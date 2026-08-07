"""
评分拉伸算子 — 部署端测试（margin 折叠）

覆盖：logit(sigmoid(m))≡m 恒等式 / 双后端打分一致 / assert_consistent 分数域自适应 /
      JSON 序列化保留 score 元数据 / 旧格式向后兼容 / 无拉伸概率契约（零回归）/
      proto round-trip 自动获得拉伸分
"""

import numpy as np
import pandas as pd
import pytest

from risk_ml import RiskPipeline, PdoScoreScaler
from risk_ml.encoding import BinnerWoeEncoder
from risk_ml.estimator import RiskXGBClassifier
from risk_ml.feature_selection import IVSelector, CorrelationSelector
from risk_ml.preprocessing import FeatureCleaner
from risk_ml.online_deploy import (
    PipelineParser,
    assert_consistent,
    DeployPipeline,
)
from risk_ml.online_deploy.demo_deploy import make_data


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def scaled_pipe():
    """带 PdoScoreScaler 的全链路 pipeline（600/1:50/50）。"""
    df = make_data(n=400, seed=7)
    X = df.drop(columns=["y"])
    y = df["y"]
    pipe = RiskPipeline(
        [
            ("cleaner", FeatureCleaner(sentinels=[-999])),
            ("binner_woe", BinnerWoeEncoder(max_bins=6)),
            ("iv_selector", IVSelector(iv_threshold=0.02)),
            ("corr_selector", CorrelationSelector(corr_threshold=0.8)),
            ("xgb", RiskXGBClassifier(n_estimators=20, max_depth=3, eval_metric="auc")),
        ],
        score_scaler=PdoScoreScaler(base_score=600.0, base_p=1 / 51, pdo=50.0),
    )
    pipe.fit(X, y)
    return pipe, X, y


@pytest.fixture(scope="module")
def plain_pipe():
    """无拉伸 pipeline（对照：概率契约零回归）。"""
    df = make_data(n=400, seed=7)
    X = df.drop(columns=["y"])
    y = df["y"]
    pipe = RiskPipeline([
        ("cleaner", FeatureCleaner(sentinels=[-999])),
        ("binner_woe", BinnerWoeEncoder(max_bins=6)),
        ("iv_selector", IVSelector(iv_threshold=0.02)),
        ("corr_selector", CorrelationSelector(corr_threshold=0.8)),
        ("xgb", RiskXGBClassifier(n_estimators=20, max_depth=3, eval_metric="auc")),
    ])
    pipe.fit(X, y)
    return pipe, X, y


# ============================================================
# 数学恒等式
# ============================================================

class TestMarginFoldIdentity:
    def test_logit_sigmoid_identity(self):
        """logit(sigmoid(m)) ≡ m：评分拉伸可直接折叠进模型 margin。"""
        m = np.linspace(-8.0, 8.0, 1001)
        p = 1.0 / (1.0 + np.exp(-m))
        logit = np.log(p) - np.log1p(-p)
        np.testing.assert_allclose(logit, m, atol=1e-9)

    def test_affine_fold_matches_sequential(self):
        """offset + scale·logit(sigmoid(m)) ≈ offset + scale·m（部署只快不慢）。"""
        offset, factor = 600.0, 50.0 / np.log(2.0)
        m = np.linspace(-6.0, 6.0, 501)
        p = 1.0 / (1.0 + np.exp(-m))
        sequential = offset - factor * (np.log(p) - np.log1p(-p))
        folded = offset - factor * m
        np.testing.assert_allclose(folded, sequential, atol=1e-6)


# ============================================================
# 双后端打分一致
# ============================================================

class TestDeployScoring:
    @pytest.mark.parametrize("backend", ["m2cgen", "onnx"])
    def test_score_matches_predict_score(self, scaled_pipe, backend):
        pipe, X, _ = scaled_pipe
        deploy = PipelineParser(backend=backend).compile_pipeline(pipe)
        rows = X.iloc[:50].to_dict("records")
        X_true = pd.DataFrame(rows, columns=deploy.feature_names_in_)
        y_truth = pipe.predict_score(X_true)
        y_deploy = deploy.score_batch(rows)
        assert y_deploy.shape == (50,)
        np.testing.assert_allclose(y_deploy, y_truth, rtol=0, atol=1.0)
        # 分数域量级（风险分量级，不在 [0,1] 概率区间），且方向可解释：低分=高风险
        assert 50.0 < y_deploy.min() and y_deploy.max() < 1000.0
        p = pipe.predict_proba(X)[:, 1]
        riskiest = X.iloc[p.argmax()]   # 概率最高 = 最高风险 → 最低分
        safest = X.iloc[p.argmin()]     # 概率最低 = 最低风险 → 最高分
        assert deploy.score(riskiest.to_dict()) < deploy.score(safest.to_dict())

    @pytest.mark.parametrize("backend", ["m2cgen", "onnx"])
    def test_assert_consistent_score_domain(self, scaled_pipe, backend):
        pipe, X, _ = scaled_pipe
        deploy = PipelineParser(backend=backend).compile_pipeline(pipe)
        # 有 scaler → 真值=风险分、atol 放宽到 1.0（checker 内部自适应）
        r = assert_consistent(pipe, deploy, X=X, atol=1e-4)
        assert r["n_fail"] == 0
        assert r["max_diff"] < 1.0

    @pytest.mark.parametrize("backend", ["m2cgen", "onnx"])
    def test_describe_shows_scaler(self, scaled_pipe, backend):
        pipe, X, _ = scaled_pipe
        deploy = PipelineParser(backend=backend).compile_pipeline(pipe)
        assert "score_scaler" in deploy.model_op.describe()


# ============================================================
# 序列化
# ============================================================

class TestSerialization:
    @pytest.mark.parametrize("backend", ["m2cgen", "onnx"])
    def test_json_roundtrip_preserves_score(self, scaled_pipe, backend):
        pipe, X, _ = scaled_pipe
        deploy = PipelineParser(backend=backend).compile_pipeline(pipe)
        d = deploy.to_dict()
        s = d["model"]["score"]
        assert s is not None
        assert s["higher_is_safer"] is True
        assert s["offset"] == pytest.approx(317.807, abs=1e-2)   # base_score=600, base_p=1/51
        assert s["factor"] == pytest.approx(50.0 / np.log(2.0), abs=1e-9)
        deploy2 = DeployPipeline.from_dict(d)
        assert deploy2.model_op.score_meta is not None
        rows = X.iloc[:30].to_dict("records")
        np.testing.assert_allclose(deploy2.score_batch(rows), deploy.score_batch(rows), atol=1e-6)

    @pytest.mark.parametrize("backend", ["m2cgen", "onnx"])
    def test_old_format_backward_compat(self, plain_pipe, backend):
        """旧 JSON（无 score 键）仍能加载：无拉伸 → 输出概率。"""
        pipe, X, _ = plain_pipe
        deploy = PipelineParser(backend=backend).compile_pipeline(pipe)
        d = deploy.to_dict()
        assert "score" not in d["model"]  # 无 scaler 不写 score 字段
        deploy2 = DeployPipeline.from_dict(d)
        assert deploy2.model_op.score_meta is None
        rows = X.iloc[:30].to_dict("records")
        y_deploy = deploy2.score_batch(rows)
        assert y_deploy.min() >= 0.0 and y_deploy.max() <= 1.0
        np.testing.assert_allclose(y_deploy, pipe.predict_proba(X.iloc[:30])[:, 1], atol=1e-4)

    @pytest.mark.parametrize("backend", ["m2cgen", "onnx"])
    def test_no_scaler_probability_contract(self, plain_pipe, backend):
        """无拉伸 pipeline：deploy.score 仍是概率，assert_consistent 用原 atol 通过。"""
        pipe, X, _ = plain_pipe
        deploy = PipelineParser(backend=backend).compile_pipeline(pipe)
        r = assert_consistent(pipe, deploy, X=X, atol=1e-4)
        assert r["n_fail"] == 0
        assert r["max_diff"] < 1e-4


# ============================================================
# proto round-trip（online_deploy_proto 零改动自动获得）
# ============================================================

class TestProtoRoundtrip:
    @pytest.mark.parametrize("backend", ["m2cgen", "onnx"])
    def test_proto_roundtrip_score(self, scaled_pipe, backend):
        """折叠烘焙进产物，executor 只消费产物 → 拉伸分自动传播，零改动。"""
        from online_deploy_proto.serialize import to_proto_bytes
        from online_deploy_proto.scorer import build_engine

        pipe, X, _ = scaled_pipe
        deploy = PipelineParser(backend=backend).compile_pipeline(pipe)
        spec = to_proto_bytes(deploy)
        engine = build_engine(spec)
        X_np = X[deploy.feature_names_in_].head(40).to_numpy(dtype=float)
        y_proto = engine.score_np(X_np)
        y_truth = pipe.predict_score(X[deploy.feature_names_in_].head(40))
        assert y_proto.shape == (40,)
        np.testing.assert_allclose(y_proto, y_truth, rtol=0, atol=1.0)
