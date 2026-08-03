"""
risk_ml.online_deploy 测试套件

覆盖：np_cut_labels / CleanerOp / BinOp / WoeOp / BinWoeOp / SelectOp /
      PipelineParser 双后端编译 / 一致性校验 / 单条与批量打分 /
      JSON 序列化类型还原（±inf、int 箱索引、cat 映射键）/
      自定义算子注册（registry + to_deploy 协议）/ 异常路径 / benchmark
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sklearn.base import BaseEstimator, TransformerMixin

from risk_ml import RiskPipeline
from risk_ml.encoding import BinnerWoeEncoder
from risk_ml.estimator import RiskXGBClassifier
from risk_ml.feature_selection import CorrelationSelector, IVSelector
from risk_ml.preprocessing import FeatureCleaner

from risk_ml.online_deploy import (
    ConsistencyError,
    DeployError,
    DeployOp,
    DeployPipeline,
    PipelineParser,
    UnsupportedStepError,
    assert_consistent,
    benchmark,
    register_deploy_builder,
)
from risk_ml.online_deploy._base import json_dumps
from risk_ml.online_deploy._ops import (
    BinOp,
    BinWoeOp,
    CleanerOp,
    SelectOp,
    WoeOp,
    np_cut_labels,
)
from risk_ml.online_deploy.demo_deploy import make_data


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def trained():
    """训练一个小型 pipeline（全链路：清洗→分箱WOE→筛选→XGB）。"""
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
# np_cut_labels — pd.cut 等价
# ============================================================

class TestNpCutLabels:
    def test_basic_binning(self):
        x = np.array([0.5, 1.5, 2.5, 3.5])
        out = np_cut_labels(x, [0.0, 1.0, 2.0, 3.0, 4.0])
        assert out.tolist() == [0, 1, 2, 3]

    def test_first_edge_inclusive(self):
        # x == edges[0] 归第 0 箱（include_lowest=True）
        assert np_cut_labels(np.array([0.0]), [0.0, 1.0])[0] == 0

    def test_upper_bound_included(self):
        # right=True：x == 上界仍归当前箱
        assert np_cut_labels(np.array([1.0]), [0.0, 1.0])[0] == 0

    def test_out_of_range_returns_missing(self):
        out = np_cut_labels(np.array([-0.1, 5.0]), [0.0, 1.0, 2.0])
        assert out.tolist() == [-1, -1]

    def test_nan_returns_missing(self):
        out = np_cut_labels(np.array([np.nan, 0.5]), [0.0, 1.0, 2.0])
        assert out.tolist() == [-1, 0]


# ============================================================
# CleanerOp
# ============================================================

class TestCleanerOp:
    def test_from_step_drop_row_rejected(self):
        fake = SimpleNamespace(
            missing_strategy="drop_row", drop_columns_=[],
            sentinels=[-999], impute_values_={}, clip_bounds_={},
            outlier_action="clip",
        )
        with pytest.raises(UnsupportedStepError):
            CleanerOp.from_step(fake, ["a", "b"])

    def test_sentinel_fill_clip(self):
        op = CleanerOp(
            "c", ["a", "b"], ["a", "b"],
            sentinels=[-999], impute_values={"a": 0.0},
            clip_bounds={"b": (0.0, 10.0)}, outlier_action="clip",
        )
        out = op.transform(np.array([[5.0, -999.0], [-999.0, 20.0]]))
        # a 列有 impute：-999 → NaN → 填充 0；b 列无 impute：-999 → NaN 保持
        assert out[0, 0] == 5.0
        assert np.isnan(out[0, 1])   # b 列 sentinel → NaN（无填充）
        assert out[1, 0] == 0.0      # a 列 -999 → 填充 0
        assert out[1, 1] == 10.0     # b 列 20 → clip 到 10

    def test_set_nan_action(self):
        op = CleanerOp(
            "c", ["a"], ["a"], sentinels=[],
            impute_values={}, clip_bounds={"a": (0.0, 1.0)},
            outlier_action="set_nan",
        )
        out = op.transform(np.array([[0.5], [5.0]]))
        assert out[0, 0] == 0.5
        assert np.isnan(out[1, 0])

    def test_round_trip_inf_bounds(self):
        op = CleanerOp(
            "c", ["a"], ["a"], sentinels=[-999],
            impute_values={"a": 0.0},
            clip_bounds={"a": (-np.inf, np.inf)}, outlier_action="clip",
        )
        d = json.loads(json_dumps(op.to_dict()))  # 走完整 JSON 序列化
        assert d["clip_bounds"]["a"] == ["-Infinity", "Infinity"]
        back = CleanerOp.from_dict(d)
        assert back.clip_bounds["a"] == (-np.inf, np.inf)
        assert back.impute_values == {"a": 0.0}


# ============================================================
# BinOp / WoeOp / BinWoeOp — 序列化类型还原（核心回归）
# ============================================================

class TestSerializationTypeRecovery:
    def test_woe_map_int_keys_round_trip(self):
        op = WoeOp("w", ["a"], ["a"], {"a": {0: 0.1, 1: -0.2}})
        s = json.dumps(op.to_dict(), allow_nan=False)  # 键会变成 str
        back = WoeOp.from_dict(json.loads(s))
        assert all(isinstance(k, int) for k in back.woe_maps["a"])
        out = back.transform(np.array([[0.0], [1.0]]))
        assert out[:, 0].tolist() == [0.1, -0.2]

    def test_bin_edges_inf_round_trip(self):
        op = BinOp("b", ["a"], ["a"], {"a": [-np.inf, 0.0, np.inf]})
        s = json.dumps(op.to_dict())
        back = BinOp.from_dict(json.loads(s))
        assert back.edges["a"] == [-np.inf, 0.0, np.inf]

    def test_cat_map_numeric_keys_round_trip(self):
        op = BinOp("b", ["a"], ["a"], {"a": [-0.5, 0.5, 1.5]},
                   {"a": {0: 0, 1: 1}})
        s = json.dumps(op.to_dict())
        back = BinOp.from_dict(json.loads(s))
        assert all(isinstance(k, int) for k in back.cat_maps["a"])
        # 分类映射仍生效
        out = back.transform(np.array([[0.0], [1.0]]))
        assert out[:, 0].tolist() == [0, 1]

    def test_binwoe_full_round_trip(self):
        op = BinWoeOp(
            "bw", ["a"], ["a"],
            {"a": [-np.inf, 0.0, np.inf]},
            {"a": {0: 0.3, 1: -0.4}},
            {"a": {0: 0, 1: 1}},
        )
        back = BinWoeOp.from_dict(json.loads(json.dumps(op.to_dict())))
        assert back.edges["a"] == [-np.inf, 0.0, np.inf]
        assert all(isinstance(k, int) for k in back.woe_maps["a"])
        assert all(isinstance(k, int) for k in back.cat_maps["a"])

    def test_json_round_trip_consistency(self, trained):
        """序列化→反序列化后，类型还原必须保住打分一致性。"""
        pipe, X, _ = trained
        deploy = PipelineParser(backend="m2cgen").compile_pipeline(pipe)
        back = DeployPipeline.from_json(deploy.to_json())
        r = assert_consistent(pipe, back, X=X, atol=1e-4)
        assert r["n_fail"] == 0


# ============================================================
# 端到端：编译 / 一致性 / 打分
# ============================================================

class TestEndToEnd:
    @pytest.mark.parametrize("backend", ["m2cgen", "onnx"])
    def test_consistency_both_backends(self, trained, backend):
        pipe, X, _ = trained
        deploy = PipelineParser(backend=backend).compile_pipeline(pipe)
        r = assert_consistent(pipe, deploy, X=X, atol=1e-4)
        assert r["n_fail"] == 0
        assert r["max_diff"] < 1e-4

    @pytest.mark.parametrize("backend", ["m2cgen", "onnx"])
    def test_single_score_matches_predict_proba(self, trained, backend):
        pipe, X, _ = trained
        deploy = PipelineParser(backend=backend).compile_pipeline(pipe)
        row = X.iloc[0].to_dict()
        truth = pipe.predict_proba(pd.DataFrame([row]))[0, 1]
        assert abs(deploy.score(row) - truth) < 1e-4
        assert 0.0 <= deploy.score(row) <= 1.0

    def test_score_batch_shape_and_range(self, trained):
        pipe, X, _ = trained
        deploy = PipelineParser(backend="m2cgen").compile_pipeline(pipe)
        rows = X.iloc[:50].to_dict("records")
        p = deploy.score_batch(rows)
        assert p.shape == (50,)
        assert p.min() >= 0.0 and p.max() <= 1.0

    def test_score_single_equals_batch(self, trained):
        pipe, X, _ = trained
        deploy = PipelineParser(backend="m2cgen").compile_pipeline(pipe)
        row = X.iloc[3].to_dict()
        single = deploy.score(row)
        batch = deploy.score_batch([row])[0]
        assert single == pytest.approx(batch, abs=1e-12)

    def test_missing_key_imputed(self, trained):
        pipe, X, _ = trained
        deploy = PipelineParser(backend="m2cgen").compile_pipeline(pipe)
        full = X.iloc[0].to_dict()
        partial = {k: v for k, v in full.items() if k not in ("amount", "income")}
        # 缺失列 → NaN → cleaner 填充，不应报错
        p = deploy.score(partial)
        assert 0.0 <= p <= 1.0

    def test_describe(self, trained):
        pipe, _, _ = trained
        deploy = PipelineParser(backend="m2cgen").compile_pipeline(pipe)
        s = deploy.describe()
        assert "DeployPipeline" in s
        assert "M2CgenBackend" in s

    def test_invalid_backend_rejected(self):
        with pytest.raises(ValueError):
            PipelineParser(backend="tensorflow")


# ============================================================
# 异常路径
# ============================================================

class TestUnsupported:
    def test_non_predict_proba_estimator(self, trained):
        from sklearn.linear_model import LogisticRegression
        pipe, X, y = trained
        pipe2 = RiskPipeline([
            ("cleaner", FeatureCleaner(sentinels=[-999])),
            ("lr", LogisticRegression(max_iter=200)),
        ])
        pipe2.fit(X, y)
        with pytest.raises(DeployError):
            PipelineParser().compile_pipeline(pipe2)

    def test_unsupported_step(self, trained):
        from sklearn.preprocessing import StandardScaler
        pipe, X, y = trained
        pipe2 = RiskPipeline([
            ("scale", StandardScaler()),
            ("xgb", RiskXGBClassifier(n_estimators=5, eval_metric="auc")),
        ])
        pipe2.fit(X, y)
        with pytest.raises(UnsupportedStepError):
            PipelineParser().compile_pipeline(pipe2)

    def test_non_numeric_category_rejected(self):
        fake = SimpleNamespace(
            bin_edges_={"cat": [0.0, 1.0, 2.0]},
            _categorical_cols_=["cat"],
            _cat_maps_={"cat": {"a": 0, "b": 1}},
        )
        with pytest.raises(UnsupportedStepError):
            BinOp.from_step(fake, ["cat"])

    def test_unknown_op_kind_on_load(self):
        from risk_ml.online_deploy._base import json_dumps
        d = {
            "version": 1,
            "feature_names_in_": ["a"],
            "ops": [{"kind": "nope", "name": "x",
                     "input_columns": ["a"], "output_columns": ["a"]}],
            "model": {"kind": "m2cgen", "feature_names": ["a"],
                      "base_score": 0.5, "code": "def score(x):\n    return 1.0"},
        }
        with pytest.raises(DeployError):
            DeployPipeline.from_dict(json.loads(json_dumps(d)))


# ============================================================
# 自定义算子扩展：registry + to_deploy 协议
# ============================================================

class Scale10(BaseEstimator, TransformerMixin):
    """自定义 transformer：特征 ×10。"""
    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        return X * 10


class Scale10Op(DeployOp):
    kind = "scale10"

    def __init__(self, name, columns):
        super().__init__(name, columns, columns)

    @classmethod
    def from_step(cls, step, columns, name=""):
        return cls(name or "s10", columns)

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d["input_columns"])

    def transform(self, X):
        return np.asarray(X, dtype=np.float64) * 10


class Doubler(BaseEstimator, TransformerMixin):
    """自定义 transformer：通过 to_deploy 协议部署。"""
    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        return X * 2

    def to_deploy(self, input_columns):
        return DoublerOp("dbl", input_columns)


class DoublerOp(DeployOp):
    kind = "doubler"

    def __init__(self, name, columns):
        super().__init__(name, columns, columns)

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d["input_columns"])

    def transform(self, X):
        return np.asarray(X, dtype=np.float64) * 2


class TestCustomOps:
    def test_register_deploy_builder(self, trained):
        from risk_ml.online_deploy.registry import _DEPLOY_BUILDERS
        register_deploy_builder(Scale10, Scale10Op.from_step)
        try:
            pipe, X, y = trained
            pipe2 = RiskPipeline([
                ("scale10", Scale10()),
                ("xgb", RiskXGBClassifier(n_estimators=5, eval_metric="auc")),
            ])
            pipe2.fit(X, y)
            deploy = PipelineParser().compile_pipeline(pipe2)
            assert any(op.name == "s10" for op in deploy.ops)
            assert_consistent(pipe2, deploy, X=X, atol=1e-4)
        finally:
            _DEPLOY_BUILDERS.pop(Scale10, None)

    def test_to_deploy_protocol(self, trained):
        pipe, X, y = trained
        pipe2 = RiskPipeline([
            ("dbl", Doubler()),
            ("xgb", RiskXGBClassifier(n_estimators=5, eval_metric="auc")),
        ])
        pipe2.fit(X, y)
        deploy = PipelineParser().compile_pipeline(pipe2)
        assert_consistent(pipe2, deploy, X=X, atol=1e-4)


# ============================================================
# 一致性校验失败路径 / benchmark
# ============================================================

class TestCheckerAndBenchmark:
    def test_consistency_error_raised(self):
        # 构造一个输出固定 0.9 的假部署，与真值必然不一致
        class FakeBackend:
            def __init__(self):
                self.feature_names = []

            def score(self, X):
                return np.full(len(X), 0.9)

            def describe(self):
                return "FakeBackend"

        pipe, X, y = SimpleNamespace(), None, None
        pipe.steps = []

        class FakePipe:
            def __init__(self):
                self.steps = []

            def predict_proba(self, df):
                return np.full((len(df), 2), [0.5, 0.5])

        fake_pipe = FakePipe()
        deploy = SimpleNamespace(feature_names_in_=["a"],
                                 score_batch=lambda rows: np.full(len(rows), 0.9))
        with pytest.raises(ConsistencyError):
            assert_consistent(fake_pipe, deploy, X=pd.DataFrame({"a": [1.0, 2.0, 3.0]}),
                              atol=1e-4, n_random=3)

    def test_on_fail_callback_instead_of_raise(self):
        class FakePipe:
            def __init__(self):
                self.steps = []

            def predict_proba(self, df):
                return np.full((len(df), 2), [0.5, 0.5])

        calls = []
        deploy = SimpleNamespace(feature_names_in_=["a"],
                                 score_batch=lambda rows: np.full(len(rows), 0.9))
        r = assert_consistent(FakePipe(), deploy, X=pd.DataFrame({"a": [1.0]}),
                              atol=1e-4, n_random=1, on_fail=lambda res, msg: calls.append(msg))
        assert r["n_fail"] > 0
        assert len(calls) == 1

    def test_benchmark_returns_metrics(self, trained):
        pipe, X, _ = trained
        deploy = PipelineParser(backend="m2cgen").compile_pipeline(pipe)
        b = benchmark(deploy, X, n_single=50, warmup_single=10,
                      batch_size=20, n_batch=5, warmup_batch=2)
        assert b["single_us"] > 0
        assert b["batch_ms"] > 0
        assert b["qps"] > 0
        assert b["batch_size"] == 20
