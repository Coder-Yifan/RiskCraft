"""ScoreLift 算子 — 模型分分箱表现（不含灰样本）。

每个数据集（训练集/测试集/跨时间验证集）各产出一个 SubSection，
全量数据集按月拆分产出月度 SubSection。
"""

import numpy as np

from .._base import ReportOperator, SubSection, placeholder_df
from .._scoring import compute_lift_table


class ScoreLiftOperator(ReportOperator):
    """模型分分箱表现（不含灰样本）。

    每个数据集产出一个 SubSection，全量数据集按月拆分产出月度 SubSection。

    Parameters
    ----------
    n_bins : int
        分箱数量，默认 10
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    @property
    def name(self) -> str:
        return "score_lift"

    @property
    def title(self) -> str:
        return "3.模型分分箱表现（不含灰样本）"

    def compute(self, context) -> list[SubSection]:
        datasets = context.get_datasets()
        if not datasets:
            return [SubSection(self.title, placeholder_df(
                "数据集未提供，请通过 ReportContext.data + tag_col + label_col + score_col 传入"
            ))]

        subs = []

        # 1. 整体分箱
        for cn_name, (y_true, y_score) in datasets.items():
            mask = y_true >= 0
            y_true_clean = y_true[mask]
            y_score_clean = y_score[mask]

            df = compute_lift_table(y_true_clean, y_score_clean, self.n_bins)
            subs.append(SubSection(title=cn_name, data=df))

        # 2. 月度拆分分箱（全量数据集都拆月）
        monthly_datasets = context.get_monthly_datasets(tag_val=None)
        for month_name, (y_true, y_score) in monthly_datasets.items():
            mask = y_true >= 0
            y_true_clean = y_true[mask]
            y_score_clean = y_score[mask]

            df = compute_lift_table(y_true_clean, y_score_clean, self.n_bins)
            subs.append(SubSection(title=f"月度-{month_name}", data=df))

        return subs

    @staticmethod
    def compute_lift_table(y_true, y_score, n_bins=10, baseline_score=None):
        """静态方法: 直接调用，无需构造 ReportContext。"""
        return compute_lift_table(y_true, y_score, n_bins, baseline_score)
