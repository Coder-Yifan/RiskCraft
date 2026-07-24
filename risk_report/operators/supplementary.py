"""SupplementaryOperator — 附件1-补充分析。"""

import numpy as np
import pandas as pd

from .._base import ReportOperator, ReportSectionResult, SubSection
from .._context import ReportContext
from .model_effect import ModelEffectOperator


class SupplementaryOperator(ReportOperator):
    """补充分析算子 — 附件1（增益归因/指标对比/MOB/画像）。"""

    @property
    def name(self) -> str:
        return "supplementary"

    @property
    def title(self) -> str:
        return "附件1-补充分析"

    def compute(self, context: ReportContext) -> ReportSectionResult:
        subs = []

        # 1. 增益归因
        attribution_rows = [
            {"维度": "样本", "绝对值": "", "备注": ""},
            {"维度": "标签", "绝对值": "", "备注": ""},
            {"维度": "特征", "绝对值": "", "备注": ""},
            {"维度": "算法", "绝对值": "", "备注": ""},
            {"维度": "其他", "绝对值": "", "备注": ""},
        ]
        subs.append(SubSection(
            title="1.增益归因",
            data=pd.DataFrame(attribution_rows),
            note="结论需手动填写",
        ))

        # 2. 新模型 vs 对标版本模型 指标对比
        datasets_new = {}
        datasets_old = {}

        if context.y_train is not None and context.y_score_train is not None:
            datasets_new[f"新模型_train"] = (np.asarray(context.y_train), np.asarray(context.y_score_train))
        if context.y_oot is not None and context.y_score_oot is not None:
            datasets_new[f"新模型_oot"] = (np.asarray(context.y_oot), np.asarray(context.y_score_oot))

        if context.baseline_scores:
            for ds_name, y, y_score_new in [
                ("train", context.y_train, context.y_score_train),
                ("oot", context.y_oot, context.y_score_oot),
            ]:
                if y is not None and context.baseline_scores.get(ds_name) is not None:
                    datasets_old[f"对标_{ds_name}"] = (np.asarray(y), np.asarray(context.baseline_scores[ds_name]))

        if datasets_new or datasets_old:
            all_datasets = {**datasets_new, **datasets_old}
            effect_df = ModelEffectOperator.compute_effect_table(all_datasets, context.metrics)
            subs.append(SubSection(
                title="1.指标对比",
                data=effect_df,
            ))

        # 3. 不同表现期下模型表现
        if context.mob_data:
            mob_rows = []
            for mob_name, (y_true_mob, y_score_mob) in context.mob_data.items():
                row = {"月份": mob_name}
                for m in context.metrics:
                    try:
                        row[m.name] = m.compute(np.asarray(y_true_mob), np.asarray(y_score_mob))
                    except Exception:
                        row[m.name] = 0.0
                mob_rows.append(row)
            subs.append(SubSection(
                title="2.不同表现期下模型表现",
                data=pd.DataFrame(mob_rows),
                note="结论需手动填写",
            ))
        else:
            subs.append(SubSection(
                title="2.不同表现期下模型表现",
                data=pd.DataFrame([{"说明": "MOB 数据未提供，请通过 ReportContext.mob_data 传入"}]),
            ))

        # 4. 模型画像表现
        if context.portrait_data is not None:
            subs.append(SubSection(
                title="3.模型画像表现",
                data=context.portrait_data,
                note="结论需手动填写",
            ))
        else:
            subs.append(SubSection(
                title="3.模型画像表现",
                data=pd.DataFrame([{"说明": "画像数据未提供，请通过 ReportContext.portrait_data 传入"}]),
            ))

        return ReportSectionResult(
            sheet_name=self.title,
            sub_sections=subs,
        )
