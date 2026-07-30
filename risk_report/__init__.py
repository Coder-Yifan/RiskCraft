"""risk_report — 模型报告自动产出模块（独立包）。

三种使用模式:
1. 日常单独调用: ScoreLiftOperator.compute_lift_table(y_true, y_score)
2. 模块化组装: ModelReport(config=custom_config).fit(context).to_excel(...)
3. 全量报告: ModelReport().fit(context).to_excel(...)

数据输入: 单 DataFrame + tag_col + label_col（替代原 X_train/y_train/X_test/...）
"""

from ._base import ReportOperator, SubSection, placeholder_df
from ._context import ReportContext, PipelineAttributes, extract_pipeline_attributes, TAG_CN_MAP
from ._config import SheetConfig, DocumentConfig
from ._format import FormatConfig, DEFAULT_FORMAT
from ._scoring import compute_lift_table, compute_swap_analysis, compute_per_feature_ks, compute_sample_stats, compute_iv_from_data
from ._templates import DEFAULT_DOCUMENT_CONFIG
from ._excel import ExcelWriter
from .report import ModelReport
from .operators import (
    MetaInfoOperator,
    DevPurposeOperator,
    ModelAssumptionOperator,
    LabelDefinitionOperator,
    SampleSelectionOperator,
    ModelingSampleOperator,
    EffectSummaryOperator,
    VarDescriptionOperator,
    VarCleaningOperator,
    VarFilterOperator,
    VarAnalysisOperator,
    VarBinningOperator,
    ModelMethodOperator,
    ModelEffectOperator,
    ScoreLiftOperator,
    ScoreLiftGrayOperator,
    AttributionOperator,
    ModelComparisonOperator,
    MobPerformanceOperator,
    PortraitOperator,
    SwapAnalysisOperator,
    VarRangeOperator,
)

__all__ = [
    # 基类与数据模型
    "ReportOperator", "SubSection", "placeholder_df",
    # 上下文
    "ReportContext", "PipelineAttributes", "extract_pipeline_attributes", "TAG_CN_MAP",
    # 配置
    "SheetConfig", "DocumentConfig",
    # 格式
    "FormatConfig", "DEFAULT_FORMAT",
    # 计算工具
    "compute_lift_table", "compute_swap_analysis", "compute_per_feature_ks", "compute_sample_stats", "compute_iv_from_data",
    # 模板
    "DEFAULT_DOCUMENT_CONFIG",
    # Excel 写入
    "ExcelWriter",
    # 算子 (22个)
    "MetaInfoOperator",
    "DevPurposeOperator", "ModelAssumptionOperator", "LabelDefinitionOperator",
    "SampleSelectionOperator", "ModelingSampleOperator", "EffectSummaryOperator",
    "VarDescriptionOperator", "VarCleaningOperator", "VarFilterOperator", "VarAnalysisOperator",
    "VarBinningOperator",
    "ModelMethodOperator", "ModelEffectOperator",
    "ScoreLiftOperator", "ScoreLiftGrayOperator",
    "AttributionOperator", "ModelComparisonOperator", "MobPerformanceOperator", "PortraitOperator",
    "SwapAnalysisOperator",
    "VarRangeOperator",
    # 组合器
    "ModelReport",
]
