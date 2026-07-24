"""SwapInOutOperator — Swap In/Out 分析。"""

import numpy as np
import pandas as pd

from .._base import ReportOperator, ReportSectionResult, SubSection
from .._context import ReportContext
from .._scoring import compute_swap_analysis


class SwapInOutOperator(ReportOperator):
    """Swap In/Out 分析算子。

    Parameters
    ----------
    cutoff_percentiles : list[float]
        切分百分比列表，默认 [10, 20]
    """

    def __init__(self, cutoff_percentiles: list[float] = [10, 20]):
        self.cutoff_percentiles = cutoff_percentiles

    @property
    def name(self) -> str:
        return "swap_analysis"

    @property
    def title(self) -> str:
        return "Swap In/Out 分析"

    def compute(self, context: ReportContext) -> ReportSectionResult:
        sub_sections = []

        # 训练集 swap
        if context.y_train is not None and context.y_score_train is not None:
            y_true = np.asarray(context.y_train)
            y_score_new = np.asarray(context.y_score_train)
            y_score_old = context.baseline_scores.get("train") if context.baseline_scores else None
            df = compute_swap_analysis(y_true, y_score_new, y_score_old, self.cutoff_percentiles)
            sub_sections.append(SubSection(title="训练集 Swap In/Out", data=df))

        # OOT swap
        if context.y_oot is not None and context.y_score_oot is not None:
            y_true = np.asarray(context.y_oot)
            y_score_new = np.asarray(context.y_score_oot)
            y_score_old = context.baseline_scores.get("oot") if context.baseline_scores else None
            df = compute_swap_analysis(y_true, y_score_new, y_score_old, self.cutoff_percentiles)
            sub_sections.append(SubSection(title="OOT Swap In/Out", data=df))

        return ReportSectionResult(
            sheet_name=self.title,
            sub_sections=sub_sections,
        )

    @staticmethod
    def compute_swap_table(
        y_true,
        y_score_new,
        y_score_old=None,
        cutoff_percentiles: list[float] = [10, 20],
    ) -> pd.DataFrame:
        """静态方法: 直接调用，无需构造 ReportContext。"""
        return compute_swap_analysis(y_true, y_score_new, y_score_old, cutoff_percentiles)
