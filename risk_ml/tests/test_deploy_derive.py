"""
feature_derivative → 在线部署 端到端测试

验证 FeatureDerivativeTransformer 作为 pipeline 步骤可编译为 DeriveOp：
- 双后端（m2cgen / onnx）+ JSON / proto 往返后仍与 sklearn 一致
- fill_value 经转译→proto→kernel 后语义不变
- 上游删列导致表达式变量缺失 → 编译期明确报错（UnsupportedStepError）
"""

import numpy as np
import pandas as pd
import pytest

from risk_ml import RiskPipeline
from risk_ml.encoding import BinnerWoeEncoder
from risk_ml.estimator import RiskXGBClassifier
from risk_ml.feature_selection import IVSelector
from risk_ml.online_deploy import (
    PipelineParser,
    UnsupportedStepError,
    assert_consistent,
)
from risk_ml.online_deploy.demo_deploy import make_data
from risk_ml.preprocessing import FeatureCleaner, FeatureDerivativeTransformer

from online_deploy_proto.serialize import from_proto_bytes, to_proto_bytes


def _pipe(fd_kwargs=None):
    """标准 FD pipeline：衍生 → 清洗 → 分箱WOE → 筛选 → XGB。"""
    return RiskPipeline([
        ("fd", FeatureDerivativeTransformer(
            {"ratio_income": "amount/income", "income_1k": "income/1000"},
            **(fd_kwargs or {}),
        )),
        ("cleaner", FeatureCleaner(sentinels=[-999])),
        ("bin_woe", BinnerWoeEncoder(max_bins=6)),
        ("select", IVSelector(iv_threshold=0.02)),
        ("model", RiskXGBClassifier(n_estimators=100, max_depth=4)),
    ])


@pytest.fixture(scope="module")
def trained():
    df = make_data(n=2000, seed=42)
    X = df.drop(columns=["y"])
    return _pipe().fit(X, df["y"]), X


@pytest.mark.parametrize("backend", ["m2cgen", "onnx"])
class TestDeployDerive:
    def test_assert_consistent(self, trained, backend):
        pipe, X = trained
        deploy = PipelineParser(backend=backend).compile_pipeline(pipe)
        # 编译结果包含 DeriveOp，且列增长正确
        derive_ops = [op for op in deploy.ops if op.kind == "derive"]
        assert len(derive_ops) == 1
        r = assert_consistent(pipe, deploy, X=X, atol=1e-4)
        assert r["n_fail"] == 0

    def test_json_round_trip(self, trained, backend):
        pipe, X = trained
        deploy = PipelineParser(backend=backend).compile_pipeline(pipe)
        back = deploy.__class__.from_dict(deploy.to_dict())
        r = assert_consistent(pipe, back, X=X, atol=1e-4)
        assert r["n_fail"] == 0

    def test_proto_round_trip(self, trained, backend):
        pipe, X = trained
        deploy = PipelineParser(backend=backend).compile_pipeline(pipe)
        back = from_proto_bytes(to_proto_bytes(deploy))
        r = assert_consistent(pipe, back, X=X, atol=1e-4)
        assert r["n_fail"] == 0


class TestDeriveEdge:
    def test_fill_value_survives_deploy(self):
        """fill_value 经转译→proto→kernel 后语义与 sklearn pipeline 一致。"""
        df = make_data(n=800, seed=3)
        X, y = df.drop(columns=["y"]), df["y"]
        pipe = _pipe(fd_kwargs={"fill_value": 0.0}).fit(X, y)
        deploy = PipelineParser(backend="m2cgen").compile_pipeline(pipe)
        r = assert_consistent(pipe, deploy, X=X, atol=1e-4)
        assert r["n_fail"] == 0

    def test_missing_variable_after_upstream_drop(self):
        """上游删列导致表达式变量缺失 → from_step 编译期明确报错。"""
        df = make_data(n=400, seed=1)
        X, y = df.drop(columns=["y"]), df["y"]

        class DropEverything(FeatureCleaner):
            """mock：清洗把所有列都删掉（模拟上游把表达式变量删光的极端场景）。"""

            def fit(self, X, y=None):
                from risk_ml._base import validate_dataframe
                X = validate_dataframe(X)
                self.feature_names_in_ = X.columns.tolist()
                self.n_features_in_ = X.shape[1]
                self.drop_columns_ = list(X.columns)
                self.impute_values_ = {}
                self.clip_bounds_ = {}
                self.outlier_action = "clip"
                return self

        fd = FeatureDerivativeTransformer({"ratio": "amount/income"}).fit(X)
        # 手工构造：FD 后接一个把 amount/income 删光的步骤 → 但 FD 在删除之前，
        # 变量在 FD 输入中仍存在。真正缺列的场景是 FD 放在删列步骤之后——
        # 直接调 from_step 验证校验逻辑。
        from risk_ml.online_deploy._ops import DeriveOp
        from types import SimpleNamespace
        with pytest.raises(UnsupportedStepError, match="不在当前列"):
            DeriveOp.from_step(
                SimpleNamespace(expression_specs_=[
                    SimpleNamespace(
                        target="ratio", expression="amount/income",
                        variables=["amount", "income"], source="x",
                    )
                ]),
                input_columns=["age", "freq"],  # 缺 amount/income
            )

    def test_derive_op_serialize_json(self, trained):
        """DeriveOp 的 to_dict/from_dict 往返（JSON 路径）。"""
        pipe, _ = trained
        deploy = PipelineParser(backend="m2cgen").compile_pipeline(pipe)
        op = [o for o in deploy.ops if o.kind == "derive"][0]
        from risk_ml.online_deploy._ops import DeriveOp
        back = DeriveOp.from_dict(op.to_dict())
        assert back.expressions == op.expressions
        X = np.random.default_rng(0).random((5, len(op.input_columns)))
        np.testing.assert_array_equal(back.transform(X), op.transform(X))
