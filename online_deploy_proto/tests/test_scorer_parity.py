"""ProtoScorer 与 DeployPipeline 精确一致性

scorer 内核独立复刻 _ops.py，必须与 DeployPipeline 逐位一致：
在随机 / 分箱边界±eps / 缺失 / 哨兵样本上 max_diff < 1e-9。
"""

import json

import numpy as np
import pytest

from risk_ml.online_deploy import PipelineParser, assert_consistent
from risk_ml.online_deploy.checker import generate_test_rows

from online_deploy_proto import ProtoScorer, build_engine, register_scorer_kernel
from online_deploy_proto import deploy_spec_pb2 as pb
from online_deploy_proto.exceptions import ScoringError
from online_deploy_proto.serialize import to_proto_bytes


class ScorerAdapter:
    """把 ProtoScorer 适配为 assert_consistent 期望的接口。"""

    def __init__(self, scorer):
        self.scorer = scorer
        self.feature_names_in_ = scorer.feature_names_in

    def score_batch(self, rows):
        return self.scorer.score_rows(rows)


class TestScorerParity:
    def test_score_rows_exact(self, deploy, spec_bytes, trained):
        pipe, X, _ = trained
        scorer = build_engine(spec_bytes)
        rows = generate_test_rows(pipe, X, n_random=200)
        p_deploy = deploy.score_batch(rows)
        p_scorer = scorer.score_rows(rows)
        assert np.abs(p_deploy - p_scorer).max() < 1e-9

    def test_single_row(self, deploy, spec_bytes, trained):
        _, X, _ = trained
        scorer = build_engine(spec_bytes)
        for r in X.iloc[:5].to_dict("records"):
            assert abs(scorer.score(r) - deploy.score(r)) < 1e-12

    def test_assert_consistent(self, trained, spec_bytes):
        pipe, X, _ = trained
        scorer = build_engine(spec_bytes)
        r = assert_consistent(pipe, ScorerAdapter(scorer), X=X, atol=1e-4)
        assert r["n_fail"] == 0

    def test_build_engine_cached(self, spec_bytes):
        assert build_engine(spec_bytes) is build_engine(spec_bytes)

    def test_rows_to_array_non_numeric(self, spec_bytes, trained):
        """非数值 cell → NaN（与 DeployPipeline._to_array 语义一致）。"""
        _, X, _ = trained
        scorer = build_engine(spec_bytes)
        row = X.iloc[0].to_dict()
        row[X.columns[0]] = "not-a-number"
        assert scorer.score(row) >= 0.0  # 不抛错，内部 NaN→填充


# ======================================================================
# RawOp executor 内核
# ======================================================================
def _raw_spec_bytes(kind, params=None, feature="a"):
    s = pb.DeploySpec()
    s.version = 1
    s.feature_names_in.extend([feature])
    op = s.ops.add(name="rawop")
    op.input_columns.extend([feature])
    op.output_columns.extend([feature])
    op.raw.kind = kind
    op.raw.params_json = (json.dumps(params or {"factor": 10.0})).encode("utf-8")
    m = s.model.m2cgen
    m.feature_names.extend([feature])
    m.base_score = 0.5
    m.code = "def score(input):\n    return 0.5"
    return s.SerializeToString()


# ======================================================================
# DeriveOp（feature_derivative）parity：kernel_derive == DeployOp.transform
# ======================================================================
class TestDeriveParity:
    """含 feature_derivative 步骤时，scorer 内核与 DeployPipeline 逐位一致。"""

    @pytest.mark.parametrize("backend", ["m2cgen", "onnx"])
    def test_derive_scorer_matches_deploy(self, derive_trained, backend):
        pipe, X, _ = derive_trained
        deploy = PipelineParser(backend=backend).compile_pipeline(pipe)
        scorer = build_engine(to_proto_bytes(deploy))
        rows = generate_test_rows(pipe, X, n_random=200)
        p_deploy = deploy.score_batch(rows)
        p_scorer = scorer.score_rows(rows)
        assert np.abs(p_deploy - p_scorer).max() < 1e-9

    def test_derive_assert_consistent(self, derive_trained):
        pipe, X, _ = derive_trained
        spec = to_proto_bytes(PipelineParser(backend="m2cgen").compile_pipeline(pipe))
        scorer = build_engine(spec)
        r = assert_consistent(pipe, ScorerAdapter(scorer), X=X, atol=1e-4)
        assert r["n_fail"] == 0


class TestRawKernel:
    def test_unregistered_kernel_raises(self):
        with pytest.raises(ScoringError):
            build_engine(_raw_spec_bytes(kind="ghost"))

    def test_registered_kernel_scores(self):
        def builder(op_params, input_idx):
            factor = op_params.get("factor", 10.0)

            def fn(X):
                return np.asarray(X, dtype=np.float64) * factor

            return fn

        register_scorer_kernel("scale10", builder)
        scorer = build_engine(_raw_spec_bytes(kind="scale10", params={"factor": 3.0}))
        p = scorer.score_rows([{"a": 1.0}, {"a": 2.0}])
        assert (p == 0.5).all()  # 内核 ×3 后接常数模型
