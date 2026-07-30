"""MobPerformance 算子 — 3.不同表现期下模型表现。"""

import numpy as np
import pandas as pd

from .._base import ReportOperator, SubSection, placeholder_df


class MobPerformanceOperator(ReportOperator):
    """不同表现期下模型表现 — MOB表（多标签列对比）。"""

    @property
    def name(self) -> str:
        return "mob_performance"

    @property
    def title(self) -> str:
        return "3.不同表现期下模型表现"

    def compute(self, context) -> list[SubSection]:
        if context.data is None or not context.extra_labels:
            return [SubSection(self.title, placeholder_df(
                "多标签数据未提供，请通过 ReportContext.extra_labels 传入额外标签列名列表"
            ))]

        if context.score_col is None or context.score_col not in context.data.columns:
            return [SubSection(self.title, placeholder_df(
                "分数列未提供，请通过 ReportContext.score_col 或 pipeline 自动计算"
            ))]

        subs = []

        # 新模型表现
        rows_new = []
        for label_col in context.extra_labels:
            if label_col not in context.data.columns:
                continue
            datasets = context.get_datasets(label_col)
            for cn_name, (y_true, y_score) in datasets.items():
                mask = y_true >= 0
                row = {"标签列": label_col, "数据集": cn_name}
                for m in context.metrics:
                    try:
                        row[m.name] = m.compute(y_true[mask], y_score[mask])
                    except Exception:
                        row[m.name] = 0.0
                rows_new.append(row)

        if rows_new:
            subs.append(SubSection("新模型表现", pd.DataFrame(rows_new)))

        # 对标版本模型表现
        if context.baseline_score_col and context.baseline_score_col in context.data.columns:
            rows_old = []
            for label_col in context.extra_labels:
                if label_col not in context.data.columns:
                    continue
                baseline_ds = context.get_baseline_datasets(label_col)
                for cn_name, (y_true, y_score) in baseline_ds.items():
                    mask = y_true >= 0
                    row = {"标签列": label_col, "数据集": cn_name}
                    for m in context.metrics:
                        try:
                            row[m.name] = m.compute(y_true[mask], y_score[mask])
                        except Exception:
                            row[m.name] = 0.0
                    rows_old.append(row)

            if rows_old:
                subs.append(SubSection("对标版本模型表现", pd.DataFrame(rows_old)))

        return subs if subs else [SubSection(self.title, placeholder_df(
            "extra_labels 列不在数据中或无有效数据"
        ))]
