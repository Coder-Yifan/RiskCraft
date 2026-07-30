"""EffectSummary 算子 — 1.6模型效果汇总。"""

import numpy as np
import pandas as pd

from .._base import ReportOperator, SubSection, placeholder_df
from .._scoring import compute_sample_stats


class EffectSummaryOperator(ReportOperator):
    """模型效果汇总 — 多子模型指标对比表（AUC/KS/Lift）。"""

    def __init__(self, lift_percentiles: list[float] | None = None):
        self.lift_percentiles = lift_percentiles or [10]

    @property
    def name(self) -> str:
        return "effect_summary"

    @property
    def title(self) -> str:
        return "1.6模型效果汇总"

    def compute(self, context) -> list[SubSection]:
        datasets = context.get_datasets()
        if not datasets:
            return [SubSection(self.title, placeholder_df(
                "数据集未提供，请通过 ReportContext.data + tag_col + label_col + score_col 传入"
            ))]

        # 构建指标表
        from risk_ml.experiment.metrics import LiftMetric

        metrics = list(context.metrics)
        for pct in self.lift_percentiles:
            # 检查是否已有该百分位的 Lift
            has_pct = any(isinstance(m, LiftMetric) and m.percentile == pct for m in metrics)
            if not has_pct:
                metrics.append(LiftMetric(percentile=pct))

        rows = []
        for cn_name, (y_true, y_score) in datasets.items():
            mask = y_true >= 0
            y_true_clean = y_true[mask]
            y_score_clean = y_score[mask]
            stats = compute_sample_stats(y_true_clean)
            row = {
                "数据集": cn_name,
                "好": stats["goods"],
                "坏": stats["bads"],
                "总量": stats["total"],
                "坏占比": stats["bad_rate"],
            }
            for m in metrics:
                try:
                    row[m.name] = m.compute(y_true_clean, y_score_clean)
                except Exception:
                    row[m.name] = 0.0
            rows.append(row)

        # 子模型指标（如有）
        if context.sub_models:
            for sub_name, sub_info in context.sub_models.items():
                sub_label = sub_info.get("label_col", context.label_col)
                sub_score = sub_info.get("score_col")
                if sub_score and sub_score in context.data.columns:
                    sub_datasets = context.get_datasets(label_col=sub_label)
                    # 用 sub_score 替换 score_col 的逻辑: 手动提取
                    for cn_name, (y_true_sub, _) in sub_datasets.items():
                        tag_val = {"训练集": "train", "测试集": "test", "跨时间验证集": "oot"}[cn_name]
                        mask_tag = context.data[context.tag_col] == tag_val
                        subset = context.data[mask_tag]
                        if len(subset) > 0 and sub_score in subset.columns and sub_label in subset.columns:
                            y_s = subset[sub_score].values.astype(float)
                            y_t = subset[sub_label].values.astype(float)
                            m_t = y_t >= 0
                            stats = compute_sample_stats(y_t[m_t])
                            row = {
                                "数据集": f"{sub_name}_{cn_name}",
                                "好": stats["goods"], "坏": stats["bads"],
                                "总量": stats["total"], "坏占比": stats["bad_rate"],
                            }
                            for m in metrics:
                                try:
                                    row[m.name] = m.compute(y_t[m_t], y_s[m_t])
                                except Exception:
                                    row[m.name] = 0.0
                            rows.append(row)

        return [SubSection(self.title, pd.DataFrame(rows))]
