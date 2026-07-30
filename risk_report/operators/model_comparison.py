"""ModelComparison 算子 — 2.模型对比效果。"""

import numpy as np
import pandas as pd

from .._base import ReportOperator, SubSection, placeholder_df
from .._scoring import compute_sample_stats


class ModelComparisonOperator(ReportOperator):
    """模型对比效果 — 新模型 vs 对标版本指标对比表。"""

    def __init__(self, lift_percentiles: list[float] | None = None):
        self.lift_percentiles = lift_percentiles or [10]

    @property
    def name(self) -> str:
        return "model_comparison"

    @property
    def title(self) -> str:
        return "2.模型对比效果"

    def compute(self, context) -> list[SubSection]:
        datasets = context.get_datasets()
        baseline_datasets = context.get_baseline_datasets()

        if not datasets and not baseline_datasets:
            return [SubSection(self.title, placeholder_df(
                "数据集或对标模型分数未提供，请通过 ReportContext.baseline_score_col 传入"
            ))]

        from risk_ml.experiment.metrics import LiftMetric

        metrics = list(context.metrics)
        for pct in self.lift_percentiles:
            has_pct = any(isinstance(m, LiftMetric) and m.percentile == pct for m in metrics)
            if not has_pct:
                metrics.append(LiftMetric(percentile=pct))

        rows = []

        # 新模型指标
        for cn_name, (y_true, y_score) in datasets.items():
            mask = y_true >= 0
            stats = compute_sample_stats(y_true[mask])
            row = {"数据集": f"新模型_{cn_name}", "好": stats["goods"], "坏": stats["bads"], "总量": stats["total"], "坏占比": stats["bad_rate"]}
            for m in metrics:
                try:
                    row[m.name] = m.compute(y_true[mask], y_score[mask])
                except Exception:
                    row[m.name] = 0.0
            rows.append(row)

        # 对标模型指标
        for cn_name, (y_true, y_score) in baseline_datasets.items():
            mask = y_true >= 0
            stats = compute_sample_stats(y_true[mask])
            row = {"数据集": f"对标_{cn_name}", "好": stats["goods"], "坏": stats["bads"], "总量": stats["total"], "坏占比": stats["bad_rate"]}
            for m in metrics:
                try:
                    row[m.name] = m.compute(y_true[mask], y_score[mask])
                except Exception:
                    row[m.name] = 0.0
            rows.append(row)

        return [SubSection(self.title, pd.DataFrame(rows))]
