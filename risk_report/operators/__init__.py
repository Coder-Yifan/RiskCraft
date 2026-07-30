"""报告算子导出 — 22 个独立算子。

原 Sheet 级算子（ModelDesignOperator/ModelPerformanceOperator 等）已拆解为
每个标题一个算子，由 SheetConfig 组合到 sheet 中。
"""

from .meta_info import MetaInfoOperator
from .dev_purpose import DevPurposeOperator
from .model_assumption import ModelAssumptionOperator
from .label_definition import LabelDefinitionOperator
from .sample_selection import SampleSelectionOperator
from .modeling_sample import ModelingSampleOperator
from .effect_summary import EffectSummaryOperator
from .var_description import VarDescriptionOperator
from .var_cleaning import VarCleaningOperator
from .var_filter import VarFilterOperator
from .var_analysis import VarAnalysisOperator
from .var_binning import VarBinningOperator
from .model_method import ModelMethodOperator
from .model_effect import ModelEffectOperator
from .score_lift import ScoreLiftOperator
from .score_lift_gray import ScoreLiftGrayOperator
from .attribution import AttributionOperator
from .model_comparison import ModelComparisonOperator
from .mob_performance import MobPerformanceOperator
from .portrait import PortraitOperator
from .swap_analysis import SwapAnalysisOperator
from .var_range import VarRangeOperator

__all__ = [
    # 模型说明
    "MetaInfoOperator",
    # 模型设计 (1.1~1.6)
    "DevPurposeOperator",
    "ModelAssumptionOperator",
    "LabelDefinitionOperator",
    "SampleSelectionOperator",
    "ModelingSampleOperator",
    "EffectSummaryOperator",
    # 变量分析 (1~4)
    "VarDescriptionOperator",
    "VarCleaningOperator",
    "VarFilterOperator",
    "VarAnalysisOperator",
    # 变量分箱
    "VarBinningOperator",
    # 模型表现 (1~4)
    "ModelMethodOperator",
    "ModelEffectOperator",
    "ScoreLiftOperator",
    "ScoreLiftGrayOperator",
    # 补充分析 (1~4)
    "AttributionOperator",
    "ModelComparisonOperator",
    "MobPerformanceOperator",
    "PortraitOperator",
    # 模型使用方案
    "SwapAnalysisOperator",
    # 变量描述
    "VarRangeOperator",
]
