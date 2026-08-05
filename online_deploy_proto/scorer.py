"""
executor 侧 ProtoScorer — 从 proto 字节直接打分的自包含打分器

- 只依赖 numpy + protobuf + deploy_spec_pb2（+ onnxruntime 惰性），
  绝不 import risk_ml（否则触发顶层 __init__ 重依赖）。
- 解析 DeploySpec 后构建 kernel 链 + 打分引擎（ONNX session 首次打分才建）。
- RawOp（自定义算子）通过 register_scorer_kernel 注册同名内核，未注册明确报错。

一致性保证：scorer 内核与 risk_ml/online_deploy/_ops.py 逐位一致，
由 tests/test_scorer_parity.py 锁死；绝不复刻 to_dict/from_dict/_num_key 路径
（_num_key 会对原生 float 键截断，见 codec.py 注释）。
"""

import json

import numpy as np

from . import deploy_spec_pb2 as pb
from ._kernels import (
    kernel_bin,
    kernel_bin_woe,
    kernel_cleaner,
    kernel_select,
    kernel_woe,
)
from ._model import M2CgenEngine, OnnxEngine
from .exceptions import ScoringError

__version__ = "0.1.0"

# ======================================================================
# 版本门控（语义化版本比较）
# ======================================================================
def _parse_semver(v):
    """'1.2.3' → (1,2,3)；缺位补 0；解析失败抛 ScoringError。"""
    parts = [p for p in str(v).split(".")[:3]]
    while len(parts) < 3:
        parts.append("0")
    try:
        return tuple(int(p) for p in parts)
    except ValueError as e:
        raise ScoringError(f"非法语义化版本号: {v!r}") from e


def check_min_scorer_version(required, current=None):
    """required > current 时抛 ScoringError（spec 由更新版本的 driver 生成）。"""
    current = current or __version__
    if required and _parse_semver(required) > _parse_semver(current):
        raise ScoringError(
            f"部署规格要求 scorer >= {required}，当前 online_deploy_proto "
            f"{current} 无法保证语义兼容，请升级后再加载"
        )


# ======================================================================
# RawOp 内核注册表（executor 侧）
# ======================================================================
_RAW_KERNELS = {}


def register_scorer_kernel(kind, builder):
    """注册 RawOp 的 executor 侧打分内核。

    Args:
        kind: RawOp.kind（与 driver 侧 register_proto_op 的 kind 一致）
        builder: callable(op_params, input_idx) -> fn(X)->X_new
            - op_params: RawOp.params_json 解析后的 dict（含 name/input/output_columns）
            - input_idx: {列名: 列位置}（按上游数组列序）
    """
    _RAW_KERNELS[kind] = builder


# ======================================================================
# ProtoScorer
# ======================================================================
class ProtoScorer:
    """从 proto 字节解析的部署规格打分器。"""

    def __init__(self, spec_bytes):
        spec = pb.DeploySpec()
        spec.ParseFromString(spec_bytes)
        if spec.version != 1:
            raise ScoringError(f"不支持的部署格式版本: {spec.version}（当前仅支持 1）")
        check_min_scorer_version(spec.min_scorer_version)
        self.spec = spec
        self.feature_names_in = list(spec.feature_names_in)

        # 预构建 kernel 链：按上游列序推进，每算子缓存 input_idx + 参数
        current_cols = list(spec.feature_names_in)
        self._kernels = []
        for op in spec.ops:
            input_idx = {c: i for i, c in enumerate(current_cols)}
            self._kernels.append(self._build_kernel(op, input_idx))
            current_cols = list(op.output_columns)
        self._engine = self._build_engine(spec.model)

    # ------------------------------------------------------------------
    # 解析
    # ------------------------------------------------------------------
    def _build_kernel(self, op, input_idx):
        which = op.WhichOneof("op")
        name = op.name
        out = list(op.output_columns)
        if which == "cleaner":
            c = op.cleaner
            if c.outlier_action == pb.OUTLIER_UNSPECIFIED:
                raise ScoringError(f"op[{name}] CleanerOp.outlier_action 未指定")
            action = "clip" if c.outlier_action == pb.CLIP else "set_nan"
            sentinels = [float(s) for s in c.sentinels]
            impute = dict(c.impute_values)
            clip = {k: (b.lower, b.upper) for k, b in c.clip_bounds.items()}
            return lambda X: kernel_cleaner(
                X, input_idx, out, sentinels, impute, clip, action
            )
        if which == "bin":
            b = op.bin
            edges = {k: list(v.edge) for k, v in b.edges.items()}
            cat = {k: {e.key: int(e.code) for e in v.entry}
                   for k, v in b.cat_maps.items()}
            return lambda X: kernel_bin(X, input_idx, out, edges, cat)
        if which == "woe":
            w = op.woe
            woe_maps = {k: dict(v.woe) for k, v in w.woe_maps.items()}
            return lambda X: kernel_woe(X, input_idx, out, woe_maps)
        if which == "bin_woe":
            bw = op.bin_woe
            edges = {k: list(v.edge) for k, v in bw.edges.items()}
            woe_maps = {k: dict(v.woe) for k, v in bw.woe_maps.items()}
            cat = {k: {e.key: int(e.code) for e in v.entry}
                   for k, v in bw.cat_maps.items()}
            return lambda X: kernel_bin_woe(X, input_idx, out, edges, woe_maps, cat)
        if which == "select":
            return lambda X: kernel_select(X, input_idx, out)
        if which == "raw":
            raw = op.raw
            builder = _RAW_KERNELS.get(raw.kind)
            if builder is None:
                raise ScoringError(
                    f"executor 未注册 RawOp 内核: {raw.kind!r}，"
                    "请在打分前调用 register_scorer_kernel 注册"
                )
            params = json.loads(raw.params_json.decode("utf-8")) if raw.params_json else {}
            params.setdefault("name", name)
            params.setdefault("input_columns", list(op.input_columns))
            params.setdefault("output_columns", out)
            return builder(params, input_idx)
        raise ScoringError(
            f"op[{name}] 未知算子类型: {which!r}（可能是更新版本生成的 spec，"
            "请升级 online_deploy_proto）"
        )

    def _build_engine(self, model):
        which = model.WhichOneof("backend")
        if which == "onnx":
            o = model.onnx
            return OnnxEngine(o.model_bytes, o.feature_names, o.base_score)
        if which == "m2cgen":
            m = model.m2cgen
            return M2CgenEngine(m.code, m.feature_names, m.base_score)
        raise ScoringError(f"DeploySpec 缺少模型后端: {which!r}")

    # ------------------------------------------------------------------
    # 打分
    # ------------------------------------------------------------------
    def _rows_to_array(self, rows):
        """list[dict] → numpy 数组 (n, f)，缺失/非数值 → NaN。

        对照 risk_ml/online_deploy/parser.py:_to_array 逐位一致。
        """
        n = len(rows)
        arr = np.full((n, len(self.feature_names_in)), np.nan)
        for r_i, row in enumerate(rows):
            for i, c in enumerate(self.feature_names_in):
                v = row.get(c)
                if v is None:
                    continue
                try:
                    arr[r_i, i] = v
                except (TypeError, ValueError):
                    pass  # 非数值值 → NaN
        return arr

    def score_np(self, X):
        """批量打分：形状 (n, f) 的数值数组 → 正例概率 (n,)。"""
        X = np.asarray(X, dtype=np.float64)
        for fn in self._kernels:
            X = fn(X)
        return self._engine.score(X)

    def score_rows(self, rows):
        """批量打分：list[dict] → 正例概率 (n,)。"""
        return self.score_np(self._rows_to_array(rows))

    def score(self, row):
        """单条打分：dict → 正例概率。"""
        return float(self.score_rows([row])[0])


# ======================================================================
# 引擎缓存（Spark executor 每分区进程内复用）
# ======================================================================
_ENGINES = {}


def build_engine(spec_bytes):
    """构建/复用 ProtoScorer（模块级缓存，按 spec_bytes 去重）。"""
    if spec_bytes not in _ENGINES:
        _ENGINES[spec_bytes] = ProtoScorer(spec_bytes)
    return _ENGINES[spec_bytes]
