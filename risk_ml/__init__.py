"""
risk_ml — 风险建模机器学习框架
==============================

sklearn 兼容的风险建模算子工具链，提供特征清洗、分箱、WOE编码、
特征筛选、模型估计等全套风控建模组件。

典型流水线:
    Raw Features → FeatureCleaner → ChiMergeBinner → WoeEncoder
                 → IVSelector → CorrelationSelector → RiskXGBClassifier

快速开始:
    from risk_ml import RiskXGBClassifier
    clf = RiskXGBClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
"""

from ._base import RiskTransformer, RiskSelector
from .preprocessing import FeatureCleaner
from .binning import ChiMergeBinner
from .encoding import WoeEncoder, BinnerWoeEncoder
from .feature_selection import IVSelector, CorrelationSelector, PSISelector
from .estimator import RiskXGBClassifier, OptunaTuner
from .dataset import LendingClubLoader
from .experiment import (
    TimeWindow,
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
    make_experiment_grid,
    BaseMetric,
    AUCMetric,
    KSMetric,
    LiftMetric,
    DEFAULT_METRICS,
)
from .report import (
    ReportOperator,
    ReportContext,
    PipelineAttributes,
    ExcelWriter,
    FormatConfig,
    ModelReport,
    ScoreLiftOperator,
    FeatureFilterSummaryOperator,
    ModelEffectOperator,
    SwapInOutOperator,
)

__all__ = [
    "RiskTransformer",
    "RiskSelector",
    "FeatureCleaner",
    "ChiMergeBinner",
    "WoeEncoder",
    "BinnerWoeEncoder",
    "IVSelector",
    "CorrelationSelector",
    "PSISelector",
    "RiskXGBClassifier",
    "OptunaTuner",
    "LendingClubLoader",
    "TimeWindow",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentRunner",
    "make_experiment_grid",
    "BaseMetric",
    "AUCMetric",
    "KSMetric",
    "LiftMetric",
    "DEFAULT_METRICS",
    # report
    "ReportOperator",
    "ReportContext",
    "PipelineAttributes",
    "ExcelWriter",
    "FormatConfig",
    "ModelReport",
    "ScoreLiftOperator",
    "FeatureFilterSummaryOperator",
    "ModelEffectOperator",
    "SwapInOutOperator",
]
