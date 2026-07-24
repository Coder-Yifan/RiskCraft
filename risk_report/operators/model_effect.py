"""ModelEffectOperator — 模型效果指标。"""

import numpy as np
import pandas as pd

from .._base import ReportOperator, ReportSectionResult, SubSection
from .._context import ReportContext
from .._scoring import compute_sample_stats


class ModelEffectOperator(ReportOperator):
    """模型效果指标算子。

    展示各数据集的 AUC/KS/Lift 等指标。

    Parameters
    ----------
    lift_percentiles : list[float]
        Lift 百分位列表，默认 [10, 5, 2, 1]
    """

    def __init__(self, lift_percentiles: list[float] = [10, 5, 2, 1]):
        self.lift_percentiles = lift_percentiles

    @property
    def name(self) -> str:
        return "model_effect"

    @property
    def title(self) -> str:
        return "模型效果"

    def compute(self, context: ReportContext) -> ReportSectionResult:
        datasets = {}

        if context.y_train is not None and context.y_score_train is not None:
            datasets["训练集"] = (np.asarray(context.y_train), np.asarray(context.y_score_train))
        if context.y_test is not None and context.y_score_test is not None:
            datasets["测试集"] = (np.asarray(context.y_test), np.asarray(context.y_score_test))
        if context.y_oot is not None and context.y_score_oot is not None:
            datasets["跨时间验证集"] = (np.asarray(context.y_oot), np.asarray(context.y_score_oot))

        df = self.compute_effect_table(datasets, context.metrics, self.lift_percentiles)

        return ReportSectionResult(
            sheet_name=self.title,
            sub_sections=[SubSection(title="2.模型效果", data=df)],
        )

    @staticmethod
    def compute_effect_table(
        datasets: dict[str, tuple],
        metrics: list | None = None,
        lift_percentiles: list[float] = [10, 5, 2, 1],
    ) -> pd.DataFrame:
        """静态方法: 直接调用，无需构造 ReportContext。

        Parameters
        ----------
        datasets : dict
            {数据集名: (y_true, y_score)}
        metrics : list[BaseMetric] | None
            指标列表，默认 DEFAULT_METRICS
        lift_percentiles : list[float]
            Lift 百分位

        Returns
        -------
        pd.DataFrame
            模型效果表
        """
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
            stats = compute_sample_stats(y_true)
            row = {
                "数据集": name,
                "好": stats["goods"],
                "坏": stats["bads"],
                "总量": stats["total"],
                "坏占比": stats["bad_rate"],
            }
            for m in all_metrics:
                try:
                    row[m.name] = m.compute(y_true, y_score)
                except Exception:
                    row[m.name] = 0.0
            rows.append(row)

        return pd.DataFrame(rows)
