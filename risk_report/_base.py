"""报告算子基类与数据模型。

核心设计:
- ReportOperator.compute() 返回 list[SubSection]，算子与 Sheet 解耦
- SheetConfig 负责将算子结果分配到 sheet（在 _config.py 中定义）
- 缺失数据时产出占位表 + 提示文字，保持模板结构完整
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ._context import ReportContext


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


def placeholder_df(msg: str) -> pd.DataFrame:
    """生成占位表 — 缺失数据时保持模板结构完整。

    Parameters
    ----------
    msg : str
        提示文字，如 "标签定义数据未提供，请通过 ReportContext.label_col 传入"

    Returns
    -------
    pd.DataFrame
        单行单列的占位表
    """
    return pd.DataFrame([{"说明": msg}])


class ReportOperator(ABC):
    """报告算子基类。

    与 BaseMetric 模式类似: 定义 name + compute 接口，
    但输出粒度为 list[SubSection]（一个或多个 DataFrame）。
    算子与 Sheet 解耦 — SheetConfig 决定算子结果归属哪个 sheet。

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
    def compute(self, context: "ReportContext") -> list[SubSection]:
        """运行算子，产出子章节列表。

        Parameters
        ----------
        context : ReportContext
            建模上下文，包含流水线、数据集、指标等

        Returns
        -------
        list[SubSection]
            子章节列表。单 DataFrame 算子返回 1 个，
            多 DataFrame 算子返回多个。
        """
        ...
