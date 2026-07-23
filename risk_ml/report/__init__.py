"""risk_ml.report — 模型报告自动产出模块。

三种使用模式:
1. 日常单独调用: ScoreLiftOperator.compute_lift_table(y_true, y_score)
2. 模块化组装: ModelReport(operators=[...]).fit(context).to_excel(...)
3. 全量报告: ModelReport().fit(context).to_excel(...)
"""

from ._base import ReportOperator, SubSection, ReportSectionResult
from ._context import ReportContext, PipelineAttributes, extract_pipeline_attributes
from ._excel import ExcelWriter
from ._format import FormatConfig, DEFAULT_FORMAT
from ._scoring import compute_lift_table, compute_swap_analysis, compute_per_feature_ks, compute_sample_stats
from .operators import (
    MetaInfoOperator,
    ModelDesignOperator,
    VariableAnalysisOperator,
    ModelPerformanceOperator,
    SupplementaryOperator,
    UsagePlanOperator,
    VariableDescriptionOperator,
    ScoreLiftOperator,
    FeatureFilterSummaryOperator,
    ModelEffectOperator,
    SwapInOutOperator,
)
from .report import ModelReport

__all__ = [
    # 基类与数据模型
    "ReportOperator", "SubSection", "ReportSectionResult",
    # 上下文
    "ReportContext", "PipelineAttributes", "extract_pipeline_attributes",
    # Excel 写入
    "ExcelWriter",
    # 格式
    "FormatConfig", "DEFAULT_FORMAT",
    # 计算工具
    "compute_lift_table", "compute_swap_analysis", "compute_per_feature_ks", "compute_sample_stats",
    # Sheet级算子
    "MetaInfoOperator", "ModelDesignOperator", "VariableAnalysisOperator",
    "ModelPerformanceOperator", "SupplementaryOperator",
    "UsagePlanOperator", "VariableDescriptionOperator",
    # 独立算子
    "ScoreLiftOperator", "FeatureFilterSummaryOperator",
    "ModelEffectOperator", "SwapInOutOperator",
    # 组合器
    "ModelReport",
]
