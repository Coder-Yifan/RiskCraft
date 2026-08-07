"""
评分拉伸算子测试 — ScoreScaler / LogitScoreScaler / PdoScoreScaler / RiskPipeline.predict_score

覆盖：显式 offset/factor / 方向约定 / PDO 校准 / PDO↔Logit 换算一致 /
      pipeline 默认概率契约 / 带拉伸打分 / clone 保留 / 数值稳定性 / KS 单调不变
"""

import numpy as np
import pytest
from sklearn.base import clone

from risk_ml import RiskPipeline, PdoScoreScaler, LogitScoreScaler, ScoreScaler
from risk_ml.encoding import BinnerWoeEncoder
from risk_ml.estimator import RiskXGBClassifier
from risk_ml.experiment import KSMetric
from risk_ml.feature_selection import IVSelector, CorrelationSelector
from risk_ml.preprocessing import FeatureCleaner
from risk_ml.online_deploy.demo_deploy import make_data


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def trained():
    """小全链路 pipeline（清洗→分箱WOE→筛选→XGB），无 scaler。"""
    df = make_data(n=300, seed=7)
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


@pytest.fixture(scope="module")
def trained_scaled(trained):
    """同 pipeline + PdoScoreScaler（600/1:50/50）。"""
    pipe, X, y = trained
    pipe_scaled = RiskPipeline(
        pipe.steps,
        score_scaler=PdoScoreScaler(base_score=600.0, base_p=1 / 51, pdo=50.0),
    )
    pipe_scaled.fit(X, y)
    return pipe_scaled, X, y


# ============================================================
# 算子单元
# ============================================================

class TestLogit:
    def test_explicit_offset_factor(self):
        # factor=72.1348 = 50/ln2：odds 翻倍 +50
        s = LogitScoreScaler(offset=600.0, factor=50.0 / np.log(2.0))
        out = s([0.5, 0.9])  # p=0.5 → 600；p=0.9（高风险）→ 更低分
        assert out[0] == pytest.approx(600.0, abs=1e-9)
        assert out[1] < 600.0
        assert np.all(np.diff(out) < 0)  # 单调递减（higher_is_safer 默认）

    def test_direction_higher_is_riskier(self):
        s = LogitScoreScaler(offset=600.0, factor=50.0, higher_is_safer=False)
        out = s([0.5, 0.9])
        assert out[0] == pytest.approx(600.0, abs=1e-9)
        assert out[1] > 600.0  # 风险越高分越高（少数内部评分）

    def test_input_validation(self):
        s = LogitScoreScaler()
        with pytest.raises(ValueError):
            s([[0.5, 0.6]])  # 二维
        with pytest.raises(ValueError):
            s([-0.1, 0.5])  # 超出 [0,1]
        with pytest.raises(ValueError):
            s([0.5, 1.5])


class TestPdo:
    def test_calibration_anchor(self):
        s = PdoScoreScaler(base_score=600.0, base_p=1 / 51, pdo=50.0)
        assert s([1 / 51])[0] == pytest.approx(600.0, abs=1e-9)

    def test_odds_doubling_halfing(self):
        s = PdoScoreScaler(base_score=600.0, base_p=1 / 51, pdo=50.0)
        # odds_good 翻倍（1/51→1/101）→ +pdo；减半（1/51→1/26）→ -pdo
        assert s([1 / 101])[0] == pytest.approx(650.0, abs=1e-9)
        assert s([1 / 26])[0] == pytest.approx(550.0, abs=1e-9)
        # 单调：p 越大分越低
        assert np.all(np.diff(s([1 / 101, 1 / 51, 1 / 26])) < 0)

    def test_pdo_matches_logit(self):
        base_p, pdo = 1 / 51, 50.0
        factor = pdo / np.log(2.0)
        offset = 600.0 - factor * np.log((1 - base_p) / base_p)
        a = PdoScoreScaler(base_score=600.0, base_p=base_p, pdo=pdo)
        b = LogitScoreScaler(offset=offset, factor=factor)
        p = np.linspace(0.01, 0.99, 50)
        np.testing.assert_allclose(a(p), b(p), rtol=0, atol=1e-9)

    def test_param_validation(self):
        with pytest.raises(ValueError):
            PdoScoreScaler(base_p=0.0)
        with pytest.raises(ValueError):
            PdoScoreScaler(base_p=1.0)
        with pytest.raises(ValueError):
            PdoScoreScaler(pdo=0.0)


class TestBase:
    def test_abstract(self):
        # 基类不能实例化，子类必须实现 transform
        with pytest.raises(TypeError):
            ScoreScaler()

    def test_subclass_extensible(self):
        class DoubleScaler(ScoreScaler):
            def transform(self, p):
                return 2 * self._validate_p(p)

        assert DoubleScaler()([0.25]).tolist() == [0.5]


# ============================================================
# pipeline 打分
# ============================================================

class TestPipeline:
    def test_default_proba_contract(self, trained):
        pipe, X, _ = trained
        s = pipe.predict_score(X)
        p = pipe.predict_proba(X)[:, 1]
        np.testing.assert_allclose(s, p, rtol=0, atol=1e-12)
        assert s.min() >= 0.0 and s.max() <= 1.0

    def test_with_scaler_stretched(self, trained_scaled):
        pipe, X, _ = trained_scaled
        s = pipe.predict_score(X)
        scaler = pipe.score_scaler
        p = pipe.predict_proba(X)[:, 1]
        np.testing.assert_allclose(s, scaler.transform(p), rtol=0, atol=1e-9)
        # 拉伸后超出 [0,1]，量级落在评分卡区间
        assert s.min() < 0.0 or s.max() > 1.0
        assert s.mean() > 100.0
        # 高风险样本（概率高）分数低：方向一致
        risky = X.iloc[p.argmax()]
        safe = X.iloc[p.argmin()]
        assert pipe.predict_score(risky.to_frame().T)[0] < pipe.predict_score(safe.to_frame().T)[0]

    def test_clone_roundtrip(self, trained_scaled):
        pipe, X, y = trained_scaled
        cloned = clone(pipe)
        assert cloned.score_scaler is not None
        cloned.fit(X, y)
        np.testing.assert_allclose(cloned.predict_score(X), pipe.predict_score(X), atol=1e-6)

    def test_numerical_stability(self):
        s = PdoScoreScaler(base_score=600.0, base_p=1 / 51, pdo=50.0)
        out = s([0.0, 1.0])  # p=0/1 裁剪，不得 inf/nan
        assert np.isfinite(out).all()


# ============================================================
# KS 单调不变
# ============================================================

class TestKSMetric:
    def test_ks_invariant_to_stretch(self, trained_scaled):
        pipe, X, y = trained_scaled
        p = pipe.predict_proba(X)[:, 1]
        s = pipe.predict_score(X)
        ks_p = KSMetric().compute(y, p)
        ks_s = KSMetric().compute(y, s)
        # KS 基于排序/分箱，对单调拉伸近似不变（数据范围自适应分箱）
        assert abs(ks_p - ks_s) < 0.05
