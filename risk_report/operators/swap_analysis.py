"""SwapAnalysis 算子 — swap in/out 对比。"""

import numpy as np
import pandas as pd

from .._base import ReportOperator, SubSection, placeholder_df
from .._scoring import compute_swap_analysis
from .._context import TAG_CN_MAP


class SwapAnalysisOperator(ReportOperator):
    """swap in/out 对比 — 10%/20%切分混淆矩阵。"""

    def __init__(self, cutoff_percentiles: list[float] | None = None):
        self.cutoff_percentiles = cutoff_percentiles or [10, 20]

    @property
    def name(self) -> str:
        return "swap_analysis"

    @property
    def title(self) -> str:
        return "swap in/out 对比"

    def compute(self, context) -> list[SubSection]:
        datasets = context.get_datasets()
        baseline_datasets = context.get_baseline_datasets()

        if not datasets:
            return [SubSection(self.title, placeholder_df(
                "数据集未提供，请通过 ReportContext.data + tag_col + label_col + score_col 传入"
            ))]

        subs = []

        for cn_name, (y_true, y_score) in datasets.items():
            mask = y_true >= 0
            y_true_clean = y_true[mask]
            y_score_clean = y_score[mask]

            y_score_old = None
            if cn_name in baseline_datasets:
                bl_y_true, bl_y_score = baseline_datasets[cn_name]
                bl_mask = bl_y_true >= 0
                y_score_old = bl_y_score[bl_mask]

            # 每个切分比例单独一个 SubSection
            for pct in self.cutoff_percentiles:
                df = compute_swap_analysis(
                    y_true_clean, y_score_clean, y_score_old, [pct],
                )
                subs.append(SubSection(title=f"{cn_name} {pct}%切分", data=df))

        return subs

    @staticmethod
    def compute_swap_table(
        y_true,
        y_score_new,
        y_score_old=None,
        cutoff_percentiles: list[float] = [10, 20],
    ) -> pd.DataFrame:
        """静态方法: 直接调用，无需构造 ReportContext。"""
        return compute_swap_analysis(y_true, y_score_new, y_score_old, cutoff_percentiles)
