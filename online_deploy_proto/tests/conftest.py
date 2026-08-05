"""online_deploy_proto 测试共享 fixture"""

import pytest

from risk_ml import RiskPipeline
from risk_ml.encoding import BinnerWoeEncoder
from risk_ml.estimator import RiskXGBClassifier
from risk_ml.feature_selection import CorrelationSelector, IVSelector
from risk_ml.preprocessing import FeatureCleaner
from risk_ml.online_deploy import PipelineParser
from risk_ml.online_deploy.demo_deploy import make_data

from online_deploy_proto.serialize import to_proto_bytes


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


@pytest.fixture(scope="module", params=["m2cgen", "onnx"])
def deploy(request, trained):
    pipe, _, _ = trained
    return PipelineParser(backend=request.param).compile_pipeline(pipe)


@pytest.fixture(scope="module")
def spec_bytes(deploy):
    return to_proto_bytes(deploy)
