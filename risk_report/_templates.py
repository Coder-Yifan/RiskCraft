"""标准模板 — 默认 DocumentConfig（8 Sheet / 22 算子）。

对应标准模板: 模型开发文档_标准化_20260604.xlsx

用户可:
1. 直接使用 DEFAULT_DOCUMENT_CONFIG 产出完整报告
2. 自定义 DocumentConfig（选择部分算子、添加自定义算子）
3. 替换特定算子（如自定义 ScoreLiftOperator 的分箱数）
"""

from ._config import DocumentConfig, SheetConfig

# 算子类延迟导入（避免循环依赖，用户可选择性导入）
from .operators.meta_info import MetaInfoOperator
from .operators.dev_purpose import DevPurposeOperator
from .operators.model_assumption import ModelAssumptionOperator
from .operators.label_definition import LabelDefinitionOperator
from .operators.sample_selection import SampleSelectionOperator
from .operators.modeling_sample import ModelingSampleOperator
from .operators.effect_summary import EffectSummaryOperator
from .operators.var_description import VarDescriptionOperator
from .operators.var_cleaning import VarCleaningOperator
from .operators.var_filter import VarFilterOperator
from .operators.var_analysis import VarAnalysisOperator
from .operators.var_binning import VarBinningOperator
from .operators.model_method import ModelMethodOperator
from .operators.model_effect import ModelEffectOperator
from .operators.score_lift import ScoreLiftOperator
from .operators.score_lift_gray import ScoreLiftGrayOperator
from .operators.attribution import AttributionOperator
from .operators.model_comparison import ModelComparisonOperator
from .operators.mob_performance import MobPerformanceOperator
from .operators.portrait import PortraitOperator
from .operators.swap_analysis import SwapAnalysisOperator
from .operators.var_range import VarRangeOperator


DEFAULT_DOCUMENT_CONFIG = DocumentConfig(sheets=[
    SheetConfig("模型说明", [
        MetaInfoOperator(),
    ]),
    SheetConfig("1.模型设计", [
        DevPurposeOperator(),
        ModelAssumptionOperator(),
        LabelDefinitionOperator(),
        SampleSelectionOperator(),
        ModelingSampleOperator(),
        EffectSummaryOperator(),
    ]),
    SheetConfig("2.变量分析", [
        VarDescriptionOperator(),
        VarCleaningOperator(),
        VarFilterOperator(),
        VarAnalysisOperator(),
    ]),
    SheetConfig("附件-变量分箱", [
        VarBinningOperator(),
    ]),
    SheetConfig("3.模型表现", [
        ModelMethodOperator(),
        ModelEffectOperator(),
        ScoreLiftOperator(),
        ScoreLiftGrayOperator(),
    ]),
    SheetConfig("附件1-补充分析", [
        AttributionOperator(),
        ModelComparisonOperator(),
        MobPerformanceOperator(),
        PortraitOperator(),
    ]),
    SheetConfig("附件2-模型使用方案", [
        SwapAnalysisOperator(),
    ]),
    SheetConfig("附件3-变量描述", [
        VarRangeOperator(),
    ]),
])
