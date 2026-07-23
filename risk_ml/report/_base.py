"""报告算子基类与数据模型。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ._context import ReportContext
    from ._format import FormatConfig


@dataclass
class SubSection:
    """子章节: Excel sheet 内的一个独立分析块。

    Parameters
    ----------
    title : str
        子章节标题，如 "1.4 样本选择"
    data : pd.DataFrame
        该子章节的数据表
    note : str
        备注说明（可选）
    """

    title: str
    data: pd.DataFrame
    note: str = ""


@dataclass
class ReportSectionResult:
    """报告章节结果: 对应一个 Excel sheet。

    Parameters
    ----------
    sheet_name : str
        Excel sheet 名，如 "2.变量分析"
    sub_sections : list[SubSection]
        该 sheet 内的子章节列表
    format_config : FormatConfig | None
        格式化配置（可选，覆盖全局默认）
    """

    sheet_name: str
    sub_sections: list[SubSection] = field(default_factory=list)
    format_config: "FormatConfig | None" = None


class ReportOperator(ABC):
    """报告算子基类。

    与 BaseMetric 模式类似: 定义 name + compute 接口，
    但输出粒度为 DataFrame（而非 float）。
    不使用 fit/transform 周期，算子是一次性计算生产者。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """算子英文名，如 'score_lift'。"""
        ...

    @property
    @abstractmethod
    def title(self) -> str:
        """算子中文标题，如 "模型分分箱表现"。"""
        ...

    @abstractmethod
    def compute(self, context: "ReportContext") -> ReportSectionResult:
        """运行算子，产出报告章节结果。

        Parameters
        ----------
        context : ReportContext
            建模上下文，包含流水线、数据集、指标等

        Returns
        -------
        ReportSectionResult
            含 sheet_name 和 sub_sections 的结果对象
        """
        ...
