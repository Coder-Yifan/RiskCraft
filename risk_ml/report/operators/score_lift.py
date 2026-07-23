"""ScoreLiftOperator — 模型分分箱表现（最常用的独立算子）。"""

import numpy as np

from .._base import ReportOperator, ReportSectionResult, SubSection
from .._context import ReportContext
from .._scoring import compute_lift_table


class ScoreLiftOperator(ReportOperator):
    """模型分分箱表现算子。

    日常监控最常用: 可通过 compute() + context 或
    静态方法 compute_lift_table() 直接调用。

    Parameters
    ----------
    n_bins : int
        分箱数量，默认 10
    score_name : str
        分数名称标签，默认 "score"
    """

    def __init__(self, n_bins: int = 10, score_name: str = "score"):
        self.n_bins = n_bins
        self.score_name = score_name

    @property
    def name(self) -> str:
        return "score_lift"

    @property
    def title(self) -> str:
        return "模型分分箱表现"

    def compute(self, context: ReportContext) -> ReportSectionResult:
        """运行算子，产出模型分分箱表现表。"""
        sub_sections = []

        # 训练集
        if context.y_train is not None and context.y_score_train is not None:
            y_true = np.asarray(context.y_train)
            y_score = np.asarray(context.y_score_train)
            baseline = context.baseline_scores.get("train") if context.baseline_scores else None
            df = compute_lift_table(y_true, y_score, self.n_bins, baseline)
            sub_sections.append(SubSection(title="TRAIN", data=df))

        # 测试集
        if context.y_test is not None and context.y_score_test is not None:
            y_true = np.asarray(context.y_test)
            y_score = np.asarray(context.y_score_test)
            baseline = context.baseline_scores.get("test") if context.baseline_scores else None
            df = compute_lift_table(y_true, y_score, self.n_bins, baseline)
            sub_sections.append(SubSection(title="TEST", data=df))

        # OOT 验证集
        if context.y_oot is not None and context.y_score_oot is not None:
            y_true = np.asarray(context.y_oot)
            y_score = np.asarray(context.y_score_oot)
            baseline = context.baseline_scores.get("oot") if context.baseline_scores else None
            df = compute_lift_table(y_true, y_score, self.n_bins, baseline)
            sub_sections.append(SubSection(title="OOT", data=df))

        return ReportSectionResult(
            sheet_name=self.title,
            sub_sections=sub_sections,
        )

    @staticmethod
    def compute_lift_table(
        y_true,
        y_score,
        n_bins: int = 10,
        baseline_score=None,
    ):
        """静态方法: 直接调用，无需构造 ReportContext。

        Parameters
        ----------
        y_true : array-like
            真实标签
        y_score : array-like
            模型分数
        n_bins : int
            分箱数
        baseline_score : array-like | None
            对标模型分数

        Returns
        -------
        pd.DataFrame
            分箱表现表
        """
        return compute_lift_table(y_true, y_score, n_bins, baseline_score)
