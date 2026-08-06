"""
FeatureDerivativeTransformer 单测 — pipeline 步骤 + 部署前置契约

验证：
- 三种 expressions 输入形式归一化
- transform 结果 == feature_derivative.transform（锚定 Pandas 引擎）
- fit 前置校验：变量缺失 / 撞名拒绝 / 重复目标 / 不安全表达式
- sklearn 兼容：get_feature_names_out / 可进 Pipeline
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from feature_derivative import MissingVariableError, transform as fd_transform
from feature_derivative import UnsafeExpressionError
from sklearn.pipeline import Pipeline

from risk_ml import RiskPipeline
from risk_ml.preprocessing import FeatureDerivativeTransformer


@pytest.fixture
def df():
    return pd.DataFrame({
        "a": [1.0, 2.0, 3.0, np.nan, -1.0],
        "b": [4.0, 5.0, np.nan, 6.0, 7.0],
    })


class TestTransform:
    def test_basic(self, df):
        fd = FeatureDerivativeTransformer({
            "ratio": "a/(a+b)",
            "a_1k": "a / 1000",
        }).fit(df)
        out = fd.transform(df)
        assert list(out.columns) == ["a", "b", "ratio", "a_1k"]
        # 与 feature_derivative 锚一致
        expected = fd_transform(df, "a/(a+b)", "ratio")["ratio"]
        np.testing.assert_allclose(out["ratio"], expected, equal_nan=True)

    def test_input_forms_equivalent(self, df):
        forms = [
            {"ratio": "a/b"},
            [("ratio", "a/b")],
            [{"target": "ratio", "expression": "a/b"}],
        ]
        results = [FeatureDerivativeTransformer(f).fit(df).transform(df)["ratio"]
                   for f in forms]
        for r in results[1:]:
            np.testing.assert_array_equal(r, results[0])

    def test_does_not_mutate_input(self, df):
        before = df.copy()
        FeatureDerivativeTransformer({"r": "a/b"}).fit(df).transform(df)
        pd.testing.assert_frame_equal(df, before)

    def test_fill_value(self, df):
        fd = FeatureDerivativeTransformer({"r": "a/b"}, fill_value=1.0).fit(df)
        expected = fd_transform(df, "a/b", "r", fill_value=1.0)["r"]
        np.testing.assert_allclose(fd.transform(df)["r"], expected, equal_nan=True)

    def test_picklable(self, df):
        """pkl 参考打分链路回归：exec 函数不可 pickle，spec 只存源码即可往返。

        这是 demo_deploy_compare 抓到的真实 bug：expression_specs_ 若持有
        exec 动态函数，pickle.dump(pipe) 会抛 _pickle.PicklingError。
        """
        fd = FeatureDerivativeTransformer({"r": "a/b"}).fit(df)
        blob = pickle.dumps(fd)
        fd2 = pickle.loads(blob)
        np.testing.assert_array_equal(
            fd2.transform(df)["r"], fd.transform(df)["r"])


class TestFitValidation:
    def test_missing_variable(self, df):
        with pytest.raises(MissingVariableError):
            FeatureDerivativeTransformer({"r": "a/(a+b+c)"}).fit(df)

    def test_target_collides_with_input(self, df):
        with pytest.raises(ValueError, match="冲突"):
            FeatureDerivativeTransformer({"a": "a/b"}).fit(df)

    def test_duplicate_target(self, df):
        with pytest.raises(ValueError, match="重复"):
            FeatureDerivativeTransformer([("r", "a/b"), ("r", "a+b")]).fit(df)

    def test_unsafe_expression(self, df):
        with pytest.raises(UnsafeExpressionError):
            FeatureDerivativeTransformer({"r": "a ** 2"}).fit(df)

    def test_requires_expressions(self, df):
        with pytest.raises(ValueError, match="expressions"):
            FeatureDerivativeTransformer().fit(df)

    def test_rejects_column_order_change(self, df):
        fd = FeatureDerivativeTransformer({"r": "a/b"}).fit(df)
        with pytest.raises(ValueError, match="不一致"):
            fd.transform(df[["b", "a"]])


class TestSklearnCompat:
    def test_get_feature_names_out(self, df):
        fd = FeatureDerivativeTransformer({"r": "a/b"}).fit(df)
        assert list(fd.get_feature_names_out()) == ["a", "b", "r"]

    def test_works_in_sklearn_pipeline(self, df):
        from sklearn.impute import SimpleImputer
        pipe = Pipeline([
            ("fd", FeatureDerivativeTransformer({"r": "a/b"})),
            ("imp", SimpleImputer(strategy="median")),
        ])
        out = pipe.fit_transform(df)
        # 末步 SimpleImputer 输出 ndarray，3 列 = 原 2 列 + 衍生 r
        assert out.shape == (len(df), 3)
        assert list(pipe.named_steps["fd"].get_feature_names_out()) == ["a", "b", "r"]

    def test_works_in_risk_pipeline(self, df):
        from sklearn.impute import SimpleImputer
        pipe = RiskPipeline([
            ("fd", FeatureDerivativeTransformer({"r": "a/b"})),
            ("imp", SimpleImputer(strategy="median")),
        ])
        pipe.fit(df)
        out = pipe.transform(df)
        assert out.shape == (len(df), 3)
