"""
四类部署模块化测试 — 证明零部署改动扩展

核心主张：分箱 / 编码 / 筛选 / 估计器 四类模块只需继承各自基类
（BaseBinner / BaseEncoder / RiskSelector / RiskEstimator），
在线部署（PipelineParser → DeployOp / TreeModel）无需任何改动即可编译。

验证方式：用四类**全新自定义子类**组装 pipeline，
- 编译后算子必须是既有内置类型（bin / woe / bin_woe / select / 树模型后端）
- 与 sklearn pipeline 打分一致（assert_consistent）
- JSON / proto 往返后仍一致
"""

import json

import numpy as np
import pandas as pd
import pytest

from risk_ml import RiskPipeline
from risk_ml._base import RiskSelector, validate_dataframe
from risk_ml.binning.base_binner import BaseBinner
from risk_ml.encoding.base_encoder import BaseEncoder
from risk_ml.estimator.base_estimator import RiskEstimator
from risk_ml.online_deploy import (
    DeployPipeline,
    PipelineParser,
    UnsupportedStepError,
    assert_consistent,
)
from risk_ml.online_deploy.demo_deploy import make_data
from risk_ml.preprocessing import FeatureCleaner


# ============================================================
# 四类自定义子类（全新实现，非现有类子类化）
# ============================================================

class MedianBinner(BaseBinner):
    """自定义分箱：按中位数二分。仅实现 _bin_column。"""

    def _bin_column(self, x, y):
        med = np.nanmedian(np.asarray(x, dtype=float))
        edges = np.array([-np.inf, med, np.inf])
        return edges, ["(-inf, med]", "(med, inf)"]


class HalfWoeEncoder(BaseEncoder):
    """自定义编码：无内嵌分箱，箱 0→-0.5、箱 1→+0.5。仅实现 fit。"""

    def fit(self, X, y=None):
        X = validate_dataframe(X)
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]
        self.woe_map_ = {col: {0: -0.5, 1: 0.5} for col in X.columns}
        return self


class SignWoeEncoder(BaseEncoder):
    """自定义编码：内嵌 MedianBinner（post-fit binner_），走 BinWoeOp。"""

    def fit(self, X, y=None):
        X = validate_dataframe(X)
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]
        self.binner_ = MedianBinner().fit(X, y)
        self.bin_edges_ = self.binner_.bin_edges_
        self.bin_labels_ = self.binner_.bin_labels_
        self.woe_map_ = {col: {0: -0.5, 1: 0.5} for col in X.columns}
        return self


class FirstHalfSelector(RiskSelector):
    """自定义筛选：保留前半列。实现 fit + _get_support_mask。"""

    def fit(self, X, y=None):
        X = validate_dataframe(X)
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]
        self.mask_ = np.arange(X.shape[1]) < (X.shape[1] // 2)
        return self

    def _get_support_mask(self):
        return self.mask_


class TinyTreeEstimator(RiskEstimator):
    """自定义估计器：XGBClassifier 封装，实现 fit + to_deploy_model。"""

    def __init__(self, n_estimators=3, max_depth=2, seed=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed

    def fit(self, X, y, **fit_kwargs):
        cat_cols = self._set_feature_meta(X)
        if cat_cols:
            X = X.copy()
            for col in cat_cols:
                X[col] = X[col].astype("category")
        from xgboost import XGBClassifier

        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.seed,
        )
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def to_deploy_model(self):
        from risk_ml.online_deploy._tree_model import TreeModel

        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, "model_")
        booster = self.model_.get_booster()
        cfg = json.loads(booster.save_config())
        base_score = float(cfg["learner"]["learner_model_param"]["base_score"])
        return TreeModel.from_xgb_booster(booster, self.feature_names_in_, base_score)


# ============================================================
# Fixture
# ============================================================

@pytest.fixture(scope="module")
def df():
    return make_data(n=400, seed=7)


def build_pipe(encoder_cls=HalfWoeEncoder):
    """四类全自定义 pipeline。"""
    return RiskPipeline([
        ("cleaner", FeatureCleaner(sentinels=[-999])),
        ("bin", MedianBinner()),
        ("enc", encoder_cls()),
        ("sel", FirstHalfSelector()),
        ("model", TinyTreeEstimator()),
    ])


def build_pipe_no_selector():
    """无筛选步骤的 pipeline（校验四类各自产物时用）。"""
    return RiskPipeline([
        ("cleaner", FeatureCleaner(sentinels=[-999])),
        ("bin", MedianBinner()),
        ("enc", HalfWoeEncoder()),
        ("model", TinyTreeEstimator()),
    ])


# ============================================================
# 四类各自 → 既有内置算子
# ============================================================

class TestCustomCategoriesMapToBuiltinOps:
    def test_custom_binner_maps_to_binop(self, df):
        pipe = RiskPipeline([
            ("cleaner", FeatureCleaner(sentinels=[-999])),
            ("bin", MedianBinner()),
            ("model", TinyTreeEstimator()),
        ])
        pipe.fit(df.drop(columns=["y"]), df["y"])
        deploy = PipelineParser().compile_pipeline(pipe)
        kinds = [op.kind for op in deploy.ops]
        assert "bin" in kinds, f"自定义 BaseBinner 应编译为 BinOp，得到 {kinds}"
        assert_consistent(pipe, deploy, X=df.drop(columns=["y"]), atol=1e-4)

    def test_custom_encoder_maps_to_woeop(self, df):
        pipe = RiskPipeline([
            ("cleaner", FeatureCleaner(sentinels=[-999])),
            ("bin", MedianBinner()),
            ("enc", HalfWoeEncoder()),
            ("model", TinyTreeEstimator()),
        ])
        pipe.fit(df.drop(columns=["y"]), df["y"])
        deploy = PipelineParser().compile_pipeline(pipe)
        kinds = [op.kind for op in deploy.ops]
        assert "woe" in kinds, f"自定义 BaseEncoder 应编译为 WoeOp，得到 {kinds}"
        assert "bin_woe" not in kinds
        assert_consistent(pipe, deploy, X=df.drop(columns=["y"]), atol=1e-4)

    def test_custom_encoder_with_binner_maps_to_binwoeop(self, df):
        pipe = RiskPipeline([
            ("cleaner", FeatureCleaner(sentinels=[-999])),
            ("enc", SignWoeEncoder()),
            ("model", TinyTreeEstimator()),
        ])
        pipe.fit(df.drop(columns=["y"]), df["y"])
        deploy = PipelineParser().compile_pipeline(pipe)
        kinds = [op.kind for op in deploy.ops]
        assert "bin_woe" in kinds, f"内嵌 binner_ 的 BaseEncoder 应编译为 BinWoeOp，得到 {kinds}"
        assert_consistent(pipe, deploy, X=df.drop(columns=["y"]), atol=1e-4)

    def test_custom_selector_maps_to_selectop(self, df):
        pipe = RiskPipeline([
            ("cleaner", FeatureCleaner(sentinels=[-999])),
            ("bin", MedianBinner()),
            ("enc", HalfWoeEncoder()),
            ("sel", FirstHalfSelector()),
            ("model", TinyTreeEstimator()),
        ])
        pipe.fit(df.drop(columns=["y"]), df["y"])
        deploy = PipelineParser().compile_pipeline(pipe)
        kinds = [op.kind for op in deploy.ops]
        assert "select" in kinds, f"自定义 RiskSelector 应编译为 SelectOp，得到 {kinds}"
        assert_consistent(pipe, deploy, X=df.drop(columns=["y"]), atol=1e-4)


# ============================================================
# 四类全自定义 pipeline 端到端（双后端 + JSON/proto 往返）
# ============================================================

class TestEndToEndModular:
    @pytest.mark.parametrize("backend", ["m2cgen", "onnx"])
    def test_compile_and_parity(self, df, backend):
        pipe = build_pipe()
        pipe.fit(df.drop(columns=["y"]), df["y"])
        deploy = PipelineParser(backend=backend).compile_pipeline(pipe)
        # 全部映射为既有内置类型，部署端零改动
        assert [op.kind for op in deploy.ops] == ["cleaner", "bin", "woe", "select"]
        assert_consistent(pipe, deploy, X=df.drop(columns=["y"]), atol=1e-4)

    @pytest.mark.parametrize("backend", ["m2cgen", "onnx"])
    def test_json_round_trip(self, df, backend):
        pipe = build_pipe()
        pipe.fit(df.drop(columns=["y"]), df["y"])
        deploy = PipelineParser(backend=backend).compile_pipeline(pipe)
        deploy2 = DeployPipeline.from_json(deploy.to_json())
        # 往返后打分与 sklearn 一致
        X = df.drop(columns=["y"])
        np.testing.assert_allclose(
            deploy2.score_batch(X.to_dict("records")),
            pipe.predict_proba(X)[:, 1],
            atol=1e-4,
        )

    @pytest.mark.parametrize("backend", ["m2cgen", "onnx"])
    def test_proto_round_trip(self, df, backend):
        from online_deploy_proto.serialize import from_proto_bytes, to_proto_bytes

        pipe = build_pipe()
        pipe.fit(df.drop(columns=["y"]), df["y"])
        deploy = PipelineParser(backend=backend).compile_pipeline(pipe)
        deploy2 = from_proto_bytes(to_proto_bytes(deploy))
        X = df.drop(columns=["y"])
        np.testing.assert_allclose(
            deploy2.score_batch(X.to_dict("records")),
            pipe.predict_proba(X)[:, 1],
            atol=1e-4,
        )


# ============================================================
# 契约边界：非基类子类不可部署
# ============================================================

class TestContractBoundary:
    def test_non_estimator_rejected(self, df):
        from sklearn.linear_model import LogisticRegression

        pipe = RiskPipeline([
            ("cleaner", FeatureCleaner(sentinels=[-999])),
            ("model", LogisticRegression(max_iter=200)),
        ])
        pipe.fit(df.drop(columns=["y"]), df["y"])
        with pytest.raises(UnsupportedStepError):
            PipelineParser().compile_pipeline(pipe)

    def test_estimator_with_categorical_rejected(self, df):
        pipe = build_pipe()
        pipe.fit(df.drop(columns=["y"]), df["y"])
        # 模拟训练时包含分类特征列（parser 在部署前拒绝）
        pipe.named_steps["model"]._has_categorical_ = True
        with pytest.raises(UnsupportedStepError):
            PipelineParser().compile_pipeline(pipe)
