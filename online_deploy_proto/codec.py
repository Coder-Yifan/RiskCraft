"""
DeployOp / 模型后端 ↔ proto 直接双向转换（driver 侧）

⚠️ 设计红线：本模块必须**直接构造** DeployOp 实例，**禁止**走 to_dict/from_dict
中间层。原因：`_ops._num_key` 会对原生 float 键做 int() 截断（`_num_key(2.5)=2`），
经 dict 中转会把 JSON 序列化的类型还原 bug 带回 proto 路径；而 proto 双精度 double
键原生保真，无需任何字符串 hack。

分类键转换规则：整数值 float 键（如 2.0）还原为 int（值不变），非整数值 float 键
（如 2.5）保持 float——先判等再 int()，杜绝截断。

自定义算子（RawOp 逃生舱）：driver 侧通过 register_proto_op 注册 to_proto/from_proto，
params_json 保存 DeployOp.to_dict() 的 JSON 语义；保真由算子作者负责。
executor 侧用 register_scorer_kernel（scorer.py）注册同名打分内核。
"""

import json

from . import deploy_spec_pb2 as pb
from .exceptions import SerializationError

# 以下 import 触发 risk_ml 重依赖，仅 driver 侧可 import（executor 禁用本模块）
from risk_ml.online_deploy._base import DeployOp
from risk_ml.online_deploy._model_m2cgen import M2CgenBackend
from risk_ml.online_deploy._model_onnx import OnnxBackend
from risk_ml.online_deploy._ops import BinOp, BinWoeOp, CleanerOp, SelectOp, WoeOp


# ======================================================================
# 分类映射 helper（proto CatEntry ↔ dict，保真双类型键）
# ======================================================================
def _cat_entries_to_map(entries):
    """CatEntries → {key: code}。整数值 float 键还原 int，非整数值保持 float。"""
    m = {}
    for e in entries.entry:
        key = e.key
        if key == int(key):
            key = int(key)  # 整数值（如 2.0）→ int；先判等，杜绝 2.5→2 截断
        m[key] = int(e.code)
    return m


def _cat_map_to_entries(cat_maps):
    """{col: {key: code}} → proto BinOp/BinWoeOp 可直接写入的 CatEntries 构建函数。"""
    def build(msg):
        for col, cmap in cat_maps.items():
            entries = msg[col]
            for k, code in cmap.items():
                e = entries.entry.add()
                e.key = float(k)  # 原生 double 键，保真
                e.code = int(code)
    return build


# ======================================================================
# Op 编码
# ======================================================================
def _cleaner_to_proto(op, m):
    m.cleaner.sentinels.extend([float(s) for s in op.sentinels])
    for c, v in op.impute_values.items():
        m.cleaner.impute_values[c] = float(v)
    for c, (lo, hi) in op.clip_bounds.items():
        b = m.cleaner.clip_bounds[c]
        b.lower = float(lo)   # ±inf 原生 double，无 "Infinity" 字符串 hack
        b.upper = float(hi)
    if op.outlier_action == "clip":
        m.cleaner.outlier_action = pb.CLIP
    elif op.outlier_action == "set_nan":
        m.cleaner.outlier_action = pb.SET_NAN
    else:
        raise SerializationError(
            f"op[{op.name}] 未知 outlier_action: {op.outlier_action!r}（应为 'clip'/'set_nan'）"
        )


def _bin_to_proto(op, m):
    for c, edges in op.edges.items():
        m.bin.edges[c].edge.extend([float(e) for e in edges])
    _cat_map_to_entries(op.cat_maps)(m.bin.cat_maps)


def _woe_to_proto(op, m):
    for c, wmap in op.woe_maps.items():
        for k, v in wmap.items():
            m.woe.woe_maps[c].woe[int(k)] = float(v)


def _bin_woe_to_proto(op, m):
    for c, edges in op.edges.items():
        m.bin_woe.edges[c].edge.extend([float(e) for e in edges])
    for c, wmap in op.woe_maps.items():
        for k, v in wmap.items():
            m.bin_woe.woe_maps[c].woe[int(k)] = float(v)
    _cat_map_to_entries(op.cat_maps)(m.bin_woe.cat_maps)


def _raw_to_proto(op, m):
    codec = _PROTO_OP_CODECS.get(op.kind)
    if codec is None:
        raise SerializationError(
            f"自定义算子 {op.kind!r} 未通过 register_proto_op 注册 codec，无法 proto 序列化"
        )
    m.raw.kind = op.kind
    m.raw.params_json = codec[0](op).encode("utf-8")


def op_to_proto(op):
    """DeployOp → proto Op 消息。"""
    if not isinstance(op, DeployOp):
        raise SerializationError(f"op 必须是 DeployOp 实例，收到 {type(op).__name__}")
    m = pb.Op()
    m.name = op.name
    m.input_columns.extend(op.input_columns)
    m.output_columns.extend(op.output_columns)
    kind = op.kind
    if kind == CleanerOp.kind:
        _cleaner_to_proto(op, m)
    elif kind == BinOp.kind:
        _bin_to_proto(op, m)
    elif kind == WoeOp.kind:
        _woe_to_proto(op, m)
    elif kind == BinWoeOp.kind:
        _bin_woe_to_proto(op, m)
    elif kind == SelectOp.kind:
        m.select.SetInParent()  # 空消息置位，保证 WhichOneof 返回 "select"
    else:
        _raw_to_proto(op, m)
    return m


# ======================================================================
# Op 解码（直接构造 DeployOp，不走 from_dict）
# ======================================================================
def _cleaner_from_msg(m, name, in_cols, out_cols):
    action = m.cleaner.outlier_action
    if action == pb.OUTLIER_UNSPECIFIED:
        raise SerializationError(f"op[{name}] CleanerOp.outlier_action 未指定")
    return CleanerOp(
        name, in_cols, out_cols,
        sentinels=[float(s) for s in m.cleaner.sentinels],
        impute_values=dict(m.cleaner.impute_values),
        clip_bounds={c: (b.lower, b.upper) for c, b in m.cleaner.clip_bounds.items()},
        outlier_action="clip" if action == pb.CLIP else "set_nan",
    )


def _bin_from_msg(m, name, in_cols, out_cols):
    edges = {c: [float(e) for e in v.edge] for c, v in m.bin.edges.items()}
    cat_maps = {c: _cat_entries_to_map(v) for c, v in m.bin.cat_maps.items()}
    return BinOp(name, in_cols, out_cols, edges, cat_maps)


def _woe_from_msg(m, name, in_cols, out_cols):
    woe_maps = {c: {int(k): float(v) for k, v in wm.woe.items()}
                for c, wm in m.woe.woe_maps.items()}
    return WoeOp(name, in_cols, out_cols, woe_maps)


def _bin_woe_from_msg(m, name, in_cols, out_cols):
    edges = {c: [float(e) for e in v.edge] for c, v in m.bin_woe.edges.items()}
    woe_maps = {c: {int(k): float(v) for k, v in wm.woe.items()}
                for c, wm in m.bin_woe.woe_maps.items()}
    cat_maps = {c: _cat_entries_to_map(v) for c, v in m.bin_woe.cat_maps.items()}
    return BinWoeOp(name, in_cols, out_cols, edges, woe_maps, cat_maps)


def proto_to_op(m):
    """proto Op 消息 → DeployOp。"""
    name = m.name
    in_cols = list(m.input_columns)
    out_cols = list(m.output_columns)
    which = m.WhichOneof("op")
    if which == "cleaner":
        return _cleaner_from_msg(m, name, in_cols, out_cols)
    if which == "bin":
        return _bin_from_msg(m, name, in_cols, out_cols)
    if which == "woe":
        return _woe_from_msg(m, name, in_cols, out_cols)
    if which == "bin_woe":
        return _bin_woe_from_msg(m, name, in_cols, out_cols)
    if which == "select":
        return SelectOp(name, in_cols, out_cols)
    if which == "raw":
        codec = _PROTO_OP_CODECS.get(m.raw.kind)
        if codec is None:
            raise SerializationError(
                f"未知自定义算子 kind={m.raw.kind!r}，未注册 from_proto codec"
            )
        params = json.loads(m.raw.params_json.decode("utf-8"))
        return codec[1](params)
    raise SerializationError(
        f"op[{name}] 未知算子类型: {which!r}（可能是更新版本生成的 spec，请升级 online_deploy_proto）"
    )


# ======================================================================
# 模型后端编码/解码
# ======================================================================
def model_to_proto(model_op):
    """OnnxBackend / M2CgenBackend → proto Model。"""
    m = pb.Model()
    if isinstance(model_op, OnnxBackend):
        o = m.onnx
        o.feature_names.extend(model_op.feature_names)
        o.base_score = model_op.base_score
        o.model_bytes = model_op.model_bytes  # 免 base64，比 JSON 省 ~33%
    elif isinstance(model_op, M2CgenBackend):
        mm = m.m2cgen
        mm.feature_names.extend(model_op.feature_names)
        mm.base_score = model_op.base_score
        mm.code = model_op.code
    else:
        raise SerializationError(f"未知模型后端类型: {type(model_op).__name__}")
    return m


def proto_to_model(m):
    """proto Model → OnnxBackend / M2CgenBackend。"""
    which = m.WhichOneof("backend")
    if which == "onnx":
        o = m.onnx
        return OnnxBackend(o.model_bytes, o.feature_names, o.base_score)
    if which == "m2cgen":
        mm = m.m2cgen
        return M2CgenBackend(mm.code, mm.feature_names, mm.base_score)
    raise SerializationError(f"DeploySpec 缺少模型后端: {which!r}")


# ======================================================================
# 自定义算子 codec 注册表（driver 侧）
# ======================================================================
_PROTO_OP_CODECS = {}


def register_proto_op(kind, to_proto, from_proto):
    """注册自定义算子 ↔ proto RawOp 的编解码器。

    Args:
        kind: RawOp.kind（与 executor 侧 register_scorer_kernel 的 kind 一致）
        to_proto: callable(op: DeployOp) -> params_json(str)
        from_proto: callable(params: dict) -> DeployOp
    """
    _PROTO_OP_CODECS[kind] = (to_proto, from_proto)
