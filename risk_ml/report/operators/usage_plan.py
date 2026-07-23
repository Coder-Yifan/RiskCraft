"""UsagePlanOperator — 附件2-模型使用方案。"""

import numpy as np
import pandas as pd

from .._base import ReportOperator, ReportSectionResult, SubSection
from .._context import ReportContext
from .._scoring import compute_swap_analysis


class UsagePlanOperator(ReportOperator):
    """模型使用方案算子 — 附件2（swap in/out + cutoff）。"""

    @property
    def name(self) -> str:
        return "usage_plan"

    @property
    def title(self) -> str:
        return "附件2-模型使用方案"

    def compute(self, context: ReportContext) -> ReportSectionResult:
        subs = []

        # Swap In/Out 对比
        if context.y_train is not None and context.y_score_train is not None:
            y_true = np.asarray(context.y_train)
            y_score_new = np.asarray(context.y_score_train)
            y_score_old = context.baseline_scores.get("train") if context.baseline_scores else None

            # 10% 切分
            swap_10 = compute_swap_analysis(y_true, y_score_new, y_score_old, [10])
            # 20% 切分
            swap_20 = compute_swap_analysis(y_true, y_score_new, y_score_old, [20])

            subs.append(SubSection(title="10%切分", data=swap_10))
            subs.append(SubSection(title="20%切分", data=swap_20))

        if context.y_oot is not None and context.y_score_oot is not None:
            y_true = np.asarray(context.y_oot)
            y_score_new = np.asarray(context.y_score_oot)
            y_score_old = context.baseline_scores.get("oot") if context.baseline_scores else None

            swap_oot_10 = compute_swap_analysis(y_true, y_score_new, y_score_old, [10])
            subs.append(SubSection(title="OOT 10%切分", data=swap_oot_10))

        # Cutoff 建议
        cutoff_rows = []
        for pct in [10, 20]:
            if context.y_train is not None and context.y_score_train is not None:
                cutoff = np.percentile(np.asarray(context.y_score_train), pct)
                # 切分后坏率
                mask = np.asarray(context.y_score_train) <= cutoff
                bad_in_cutoff = (np.asarray(context.y_train)[mask] == 1).sum()
                total_in_cutoff = mask.sum()
                bad_rate = bad_in_cutoff / total_in_cutoff if total_in_cutoff > 0 else 0

                cutoff_rows.append({
                    "切分比例": f"{pct}%",
                    "cutoff分数": round(cutoff, 4),
                    "拒绝人数": total_in_cutoff,
                    "拒绝坏人": bad_in_cutoff,
                    "拒绝坏率": bad_rate,
                })

        if cutoff_rows:
            subs.append(SubSection(title="Cutoff建议", data=pd.DataFrame(cutoff_rows)))

        return ReportSectionResult(
            sheet_name=self.title,
            sub_sections=subs,
        )
