"""ScoreLiftGray 算子 — 模型分分箱表现（含灰样本）。

与 ScoreLiftOperator 类似，但 y_true 包含灰样本（Y=-1）。
"""

import numpy as np

from .._base import ReportOperator, SubSection, placeholder_df
from .._scoring import compute_lift_table


class ScoreLiftGrayOperator(ReportOperator):
    """模型分分箱表现（含灰样本）。

    将灰样本合并到各数据集的 y_true 中（Y=0好, Y=1坏, Y=-1灰）。
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    @property
    def name(self) -> str:
        return "score_lift_gray"

    @property
    def title(self) -> str:
        return "4.模型分分箱表现（含灰样本）"

    def compute(self, context) -> list[SubSection]:
        # 必须有 gray_tag 才能计算含灰报告
        if context.gray_tag is None:
            return [SubSection(self.title, placeholder_df(
                "灰样本未提供，请通过 ReportContext.gray_tag 传入"
            ))]

        datasets_with_gray = context.get_datasets_with_gray()
        if not datasets_with_gray:
            return [SubSection(self.title, placeholder_df(
                "数据集或灰样本未提供，请通过 ReportContext.data + gray_tag 传入"
            ))]

        baseline_datasets = context.get_baseline_datasets()
        subs = []

        for cn_name, (y_true, y_score) in datasets_with_gray.items():
            y_score_old = None
            # 去掉"(含灰)"后缀查找 baseline
            base_name = cn_name.replace("(含灰)", "")
            if base_name in baseline_datasets:
                bl_y_true, bl_y_score = baseline_datasets[base_name]
                # baseline 也合并灰样本
                gray_ds = context.get_gray_datasets()
                if gray_ds:
                    gray_y, gray_score = gray_ds["灰样本"]
                    bl_mask = bl_y_true >= 0
                    y_score_old = np.concatenate([bl_y_score[bl_mask], gray_score])
                else:
                    bl_mask = bl_y_true >= 0
                    y_score_old = bl_y_score[bl_mask]

            df = compute_lift_table(
                y_true, y_score,
                self.n_bins, baseline_score=y_score_old,
            )
            subs.append(SubSection(title=cn_name, data=df))

        return subs
