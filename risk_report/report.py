"""ModelReport 组合器 — 完整/部分报告编排。"""

from typing import Self

from ._base import ReportOperator, ReportSectionResult
from ._context import ReportContext
from ._excel import ExcelWriter
from ._format import FormatConfig
from .operators import (
    MetaInfoOperator,
    ModelDesignOperator,
    VariableAnalysisOperator,
    ModelPerformanceOperator,
    SupplementaryOperator,
    UsagePlanOperator,
    VariableDescriptionOperator,
)


class ModelReport:
    """模型开发报告组合器。

    支持三种使用模式:
    1. 全量报告 — 默认7个sheet级算子
    2. 模块化组装 — 选择部分算子
    3. 日常调用 — 单独调用算子的 compute() 或静态方法

    Parameters
    ----------
    operators : list[ReportOperator] | None
        报告算子列表。None 时使用 DEFAULT_OPERATORS（7个sheet级算子）。
    """

    DEFAULT_OPERATORS = [
        MetaInfoOperator(),
        ModelDesignOperator(),
        VariableAnalysisOperator(),
        ModelPerformanceOperator(),
        SupplementaryOperator(),
        UsagePlanOperator(),
        VariableDescriptionOperator(),
    ]

    def __init__(self, operators: list[ReportOperator] | None = None):
        self.operators = operators or self.DEFAULT_OPERATORS

    def fit(self, context: ReportContext) -> Self:
        """运行所有算子，存储结果。

        Parameters
        ----------
        context : ReportContext
            建模上下文

        Returns
        -------
        self
        """
        self.context = context
        self.results_: dict[str, ReportSectionResult] = {}
        for op in self.operators:
            self.results_[op.name] = op.compute(context)
        return self

    def to_excel(
        self,
        file_path: str,
        format_config: FormatConfig | None = None,
    ) -> None:
        """将所有结果写入 Excel。

        Parameters
        ----------
        file_path : str
            输出文件路径
        format_config : FormatConfig | None
            格式化配置（可选，覆盖默认）
        """
        writer = ExcelWriter(file_path, format_config=format_config)
        writer.write_report(list(self.results_.values()))
        writer.save()

    def get_section(self, name: str) -> ReportSectionResult:
        """获取指定算子的结果。

        Parameters
        ----------
        name : str
            算子 name，如 'score_lift'

        Returns
        -------
        ReportSectionResult
        """
        if name not in self.results_:
            raise KeyError(f"算子 '{name}' 不在结果中，可用: {list(self.results_.keys())}")
        return self.results_[name]
