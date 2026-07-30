"""ModelEffect 算子 — 2.模型效果。"""

import numpy as np
import pandas as pd

from .._base import ReportOperator, SubSection, placeholder_df
from .._scoring import compute_sample_stats


class ModelEffectOperator(ReportOperator):
    """模型效果 — 指标对比表（AUC/KS/Lift@10%/5%/2%/1%/PSI）。

    产出两个 SubSection:
    1. 整体指标表（训练集/测试集/跨时间验证集）
    2. 月度拆分指标表（每个数据集按月拆分）
    """

    def __init__(self, lift_percentiles: list[float] | None = None):
        self.lift_percentiles = lift_percentiles or [10, 5, 2, 1]

    @property
    def name(self) -> str:
        return "model_effect"

    @property
    def title(self) -> str:
        return "2.模型效果"

    def compute(self, context) -> list[SubSection]:
        datasets = context.get_datasets()
        if not datasets:
            return [SubSection(self.title, placeholder_df(
                "数据集未提供，请通过 ReportContext.data + tag_col + label_col + score_col 传入"
            ))]

        subs = []

        # 1. 整体指标表
        df = self.compute_effect_table(datasets, context.metrics, self.lift_percentiles)
        subs.append(SubSection(self.title, df))

        # 2. 月度拆分指标表（全量数据集都拆月）
        monthly_all = context.get_monthly_datasets(tag_val=None)
        if monthly_all:
            df_monthly = self.compute_effect_table(monthly_all, context.metrics, self.lift_percentiles)
            subs.append(SubSection("月度拆分", df_monthly))

        return subs

    @staticmethod
    def compute_effect_table(
        datasets: dict[str, tuple],
        metrics: list | None = None,
        lift_percentiles: list[float] = [10, 5, 2, 1],
    ) -> pd.DataFrame:
        """静态方法: 直接调用，无需构造 ReportContext。"""
        from risk_ml.experiment.metrics import DEFAULT_METRICS, LiftMetric

        if metrics is None:
            metrics = list(DEFAULT_METRICS)

        # 补充不同百分位的 Lift
        base_lift_pcts = set()
        for m in metrics:
            if isinstance(m, LiftMetric):
                base_lift_pcts.add(m.percentile)

        all_metrics = list(metrics)
        for pct in lift_percentiles:
            if pct not in base_lift_pcts:
                all_metrics.append(LiftMetric(percentile=pct))

        rows = []
        for name, (y_true, y_score) in datasets.items():
            y_true = np.asarray(y_true)
            y_score = np.asarray(y_score)
            # 排除灰样本
            mask = y_true >= 0
            y_true_c = y_true[mask]
            y_score_c = y_score[mask]
            stats = compute_sample_stats(y_true_c)
            row = {
                "数据集": name,
                "好": stats["goods"],
                "坏": stats["bads"],
                "总量": stats["total"],
                "坏占比": stats["bad_rate"],
            }
            for m in all_metrics:
                try:
                    row[m.name] = m.compute(y_true_c, y_score_c)
                except Exception:
                    row[m.name] = 0.0

            rows.append(row)

        return pd.DataFrame(rows)
