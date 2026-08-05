"""driver 侧 codec / serialize 测试

覆盖：双后端 proto round-trip 一致性 / ±inf·NaN 保真（无字符串 hack）/
      cat float 键 2.5 回归（锁死 _num_key 截断 bug）/ 字节确定性 /
      RawOp 往返 + 未注册报错 / version·min_scorer_version 门控
"""

import json

import numpy as np
import pytest

from risk_ml.online_deploy import assert_consistent
from risk_ml.online_deploy._base import DeployOp
from risk_ml.online_deploy._ops import BinOp, CleanerOp

from online_deploy_proto import deploy_spec_pb2 as pb
from online_deploy_proto.codec import op_to_proto, proto_to_op, register_proto_op
from online_deploy_proto.exceptions import DeploySpecError, SerializationError
from online_deploy_proto.serialize import from_proto_bytes, to_proto_bytes


class TestRoundTrip:
    def test_round_trip_bit_exact(self, deploy, trained):
        pipe, X, _ = trained
        spec = to_proto_bytes(deploy)
        back = from_proto_bytes(spec)
        rows = X.iloc[:50].to_dict("records")
        # 反序列化后打分与原始部署逐位相等
        assert (back.score_batch(rows) == deploy.score_batch(rows)).all()
        r = assert_consistent(pipe, back, X=X, atol=1e-4)
        assert r["n_fail"] == 0

    def test_onnx_proto_smaller_than_json(self, deploy):
        spec = to_proto_bytes(deploy)
        j = deploy.to_json().encode("utf-8")
        assert len(spec) <= len(j)
        if deploy.model_op.__class__.__name__ == "OnnxBackend":
            # 免 base64：onnx 后端期望显著更小（~30%）
            assert len(spec) < len(j) * 0.85

    def test_byte_determinism(self, deploy):
        # proto3 map 按 key 排序 → 同 deploy 两次序列化字节相等（可作缓存 key）
        assert to_proto_bytes(deploy) == to_proto_bytes(deploy)


class TestTypeFidelity:
    def test_inf_fidelity_no_string_hack(self):
        # proto double 原生支持 ±inf，绝不出现 JSON 的 "Infinity" 字符串
        op = BinOp("b", ["a"], ["a"], {"a": [-np.inf, 0.0, np.inf]})
        m = op_to_proto(op)
        assert b"Infinity" not in m.SerializeToString()
        back = proto_to_op(m)
        assert back.edges["a"] == [-np.inf, 0.0, np.inf]

    def test_cleaner_clip_inf_round_trip(self):
        op = CleanerOp("c", ["a"], ["a"], sentinels=[-999], impute_values={},
                       clip_bounds={"a": (-np.inf, np.inf)}, outlier_action="clip")
        back = proto_to_op(op_to_proto(op))
        assert back.clip_bounds["a"] == (-np.inf, np.inf)
        assert back.outlier_action == "clip"

    def test_cat_float_key_regression(self):
        """cat 映射非整数值 float 键（2.5）proto round-trip 必须命中。

        直接 codec 用 double 键保真 + 先判等再 int()；若误走 from_dict/_num_key，
        2.5 会被截断或还原失败 → x=2.5 落错箱，本测试锁死该回归。
        """
        op = BinOp("b", ["a"], ["a"], {"a": [-np.inf, 0.5, 1.5, np.inf]},
                   {"a": {2.5: 0, 1: 1}})
        back = proto_to_op(op_to_proto(op))
        X = np.array([[2.5], [1.0], [0.7], [np.nan]])
        assert (back.transform(X) == op.transform(X)).all()
        # 2.5→code0→箱0；1.0→code1→箱1；0.7(非键)→NaN→缺失箱；NaN→缺失箱
        assert op.transform(X)[:, 0].tolist() == [0, 1, -1, -1]

    def test_woe_float_keys_preserved(self):
        from risk_ml.online_deploy._ops import BinWoeOp
        op = BinWoeOp("bw", ["a"], ["a"],
                      {"a": [-np.inf, 0.0, np.inf]},
                      {"a": {0: 0.3, 1: -0.4}},
                      {"a": {2.5: 0, 1: 1}})
        back = proto_to_op(op_to_proto(op))
        assert back.woe_maps["a"] == {0: 0.3, 1: -0.4}
        assert back.cat_maps["a"] == {2.5: 0, 1: 1}
        X = np.array([[0.0], [2.5]])
        # assert_array_equal 视 NaN 为相等（== 对 NaN 恒 False）
        np.testing.assert_array_equal(back.transform(X), op.transform(X))


class TestVersionGates:
    def test_unknown_version_rejected(self, deploy):
        spec = pb.DeploySpec()
        spec.ParseFromString(to_proto_bytes(deploy))
        spec.version = 2
        with pytest.raises(SerializationError):
            from_proto_bytes(spec.SerializeToString())

    def test_min_scorer_version_gate(self, deploy):
        spec = pb.DeploySpec()
        spec.ParseFromString(to_proto_bytes(deploy))
        spec.min_scorer_version = "99.0.0"
        data = spec.SerializeToString()
        with pytest.raises(DeploySpecError):
            from_proto_bytes(data)
        from online_deploy_proto import build_engine
        with pytest.raises(DeploySpecError):
            build_engine(data)


# ======================================================================
# 自定义算子（RawOp 逃生舱）
# ======================================================================
class Scale10Op(DeployOp):
    kind = "scale10"

    def __init__(self, name, columns, factor=10.0):
        super().__init__(name, columns, columns)
        self.factor = float(factor)

    def transform(self, X):
        return np.asarray(X, dtype=np.float64) * self.factor

    def to_dict(self):
        d = super().to_dict()
        d["factor"] = self.factor
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d["input_columns"], factor=d.get("factor", 10.0))


class NopeOp(DeployOp):
    kind = "nope"

    def __init__(self, name, columns):
        super().__init__(name, columns, columns)

    def transform(self, X):
        return np.asarray(X, dtype=np.float64)


class TestRawOp:
    def test_custom_op_round_trip(self):
        def _to(op):
            return json.dumps(op.to_dict(), allow_nan=False)

        def _from(params):
            return Scale10Op(params["name"], params["input_columns"],
                             factor=params.get("factor", 10.0))

        register_proto_op("scale10", _to, _from)
        op = Scale10Op("s10", ["a", "b"], factor=3.0)
        back = proto_to_op(op_to_proto(op))
        assert isinstance(back, Scale10Op)
        assert back.factor == 3.0
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert (back.transform(X) == op.transform(X)).all()

    def test_unregistered_custom_op_rejected(self):
        op = NopeOp("n", ["a"])
        with pytest.raises(SerializationError):
            op_to_proto(op)
