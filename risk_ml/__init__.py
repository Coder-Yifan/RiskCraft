"""
risk_ml — 风险建模机器学习框架
==============================

sklearn 兼容的风险建模算子工具链，提供特征清洗、分箱、WOE编码、
特征筛选、模型估计等全套风控建模组件。

典型流水线:
    Raw Features → FeatureCleaner → ChiMergeBinner → WoeEncoder
                 → IVSelector → CorrelationSelector
                 → RiskXGBClassifier / RiskLGBMClassifier

快速开始:
    from risk_ml import RiskXGBClassifier
    clf = RiskXGBClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
"""

from ._base import RiskTransformer, RiskSelector
from ._pipeline import RiskPipeline
from .preprocessing import FeatureCleaner
from .binning import ChiMergeBinner
from .encoding import WoeEncoder, BinnerWoeEncoder
from .feature_selection import IVSelector, CorrelationSelector, PSISelector
from .estimator import RiskEstimator, RiskXGBClassifier, RiskLGBMClassifier, OptunaTuner
from .scoring import ScoreScaler, LogitScoreScaler, PdoScoreScaler
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
# 兼容性重导出: report 已迁移为独立包 risk_report
# 推荐直接使用 from risk_report import ...; 此重导出将在未来版本移除
# 使用 __getattr__ 懒加载避免循环导入
_REPORT_NAMES = {
    "ReportOperator", "ReportContext", "PipelineAttributes",
    "ExcelWriter", "FormatConfig", "ModelReport",
    "ScoreLiftOperator", "ModelEffectOperator", "SwapAnalysisOperator",
    "SubSection", "SheetConfig", "DocumentConfig",
    "DEFAULT_DOCUMENT_CONFIG", "placeholder_df",
}

def __getattr__(name):
    if name in _REPORT_NAMES:
        import risk_report as _rp
        return getattr(_rp, name)
    raise AttributeError(f"module 'risk_ml' has no attribute '{name}'")

__all__ = [
    "RiskPipeline",
    "RiskTransformer",
    "RiskSelector",
    "FeatureCleaner",
    "ChiMergeBinner",
    "WoeEncoder",
    "BinnerWoeEncoder",
    "IVSelector",
    "CorrelationSelector",
    "PSISelector",
    "RiskEstimator",
    "RiskXGBClassifier",
    "RiskLGBMClassifier",
    "OptunaTuner",
    "ScoreScaler",
    "LogitScoreScaler",
    "PdoScoreScaler",
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
    "ModelEffectOperator",
    "SwapAnalysisOperator",
    "SubSection",
    "SheetConfig",
    "DocumentConfig",
    "DEFAULT_DOCUMENT_CONFIG",
]
