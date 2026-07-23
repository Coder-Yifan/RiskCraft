"""ModelDesignOperator — 模型设计（1.1~1.6）。"""

import numpy as np
import pandas as pd

from .._base import ReportOperator, ReportSectionResult, SubSection
from .._context import ReportContext
from .._scoring import compute_sample_stats
from .model_effect import ModelEffectOperator


class ModelDesignOperator(ReportOperator):
    """模型设计算子 — 产出 1.1~1.6 六个子章节。"""

    @property
    def name(self) -> str:
        return "model_design"

    @property
    def title(self) -> str:
        return "1.模型设计"

    def compute(self, context: ReportContext) -> ReportSectionResult:
        subs = []

        # 1.1 模型开发目的
        subs.append(SubSection(
            title="1.1模型开发目的",
            data=pd.DataFrame([{
                "背景": context.background or "未填写",
                "应用场景": context.application or "未填写",
            }]),
        ))

        # 1.2 模型假设
        params_df = pd.DataFrame()
        if context.pipeline_attrs and context.pipeline_attrs.model_params_:
            params = context.pipeline_attrs.model_params_
            rows = [{"参数名": k, "参数值": str(v)} for k, v in params.items()]
            params_df = pd.DataFrame(rows)
        else:
            params_df = pd.DataFrame([{"参数名": "未提取", "参数值": ""}])
        subs.append(SubSection(title="1.2模型假设", data=params_df))

        # 1.3 标签定义
        label_rows = []
        for label_val, label_desc in context.label_definition.items():
            label_rows.append({"标签": f"Y={label_val}", "描述": label_desc})
        subs.append(SubSection(
            title="1.3标签定义",
            data=pd.DataFrame(label_rows),
        ))

        # 1.4 样本选择
        if context.sample_origin_distribution is not None:
            subs.append(SubSection(
                title="1.4原始样本分布",
                data=context.sample_origin_distribution,
            ))
        else:
            subs.append(SubSection(
                title="1.4原始样本分布",
                data=pd.DataFrame([{"说明": "原始样本分布数据未提供，请通过 ReportContext.sample_origin_distribution 传入"}]),
                note="数据需外部提供",
            ))

        if context.dev_sample_origin_distribution is not None:
            subs.append(SubSection(
                title="1.4开发样本分布",
                data=context.dev_sample_origin_distribution,
            ))

        # 1.5 建模样本
        sample_rows = []
        for name, y in [
            ("训练集", context.y_train),
            ("测试集", context.y_test),
            ("跨时间验证集", context.y_oot),
        ]:
            if y is not None:
                stats = compute_sample_stats(np.asarray(y), context.label_definition)
                sample_rows.append({
                    "样本集": name,
                    "好样本": stats["goods"],
                    "坏样本": stats["bads"],
                    "总量": stats["total"],
                    "坏占比": stats["bad_rate"],
                    "备注": "",
                })

        if sample_rows:
            # 总计行
            total_goods = sum(r["好样本"] for r in sample_rows)
            total_bads = sum(r["坏样本"] for r in sample_rows)
            total_n = sum(r["总量"] for r in sample_rows)
            sample_rows.append({
                "样本集": "总计",
                "好样本": total_goods,
                "坏样本": total_bads,
                "总量": total_n,
                "坏占比": total_bads / total_n if total_n > 0 else 0,
                "备注": "",
            })
            subs.append(SubSection(
                title="1.5建模样本",
                data=pd.DataFrame(sample_rows),
            ))

        # 1.6 模型效果汇总
        datasets = {}
        if context.y_train is not None and context.y_score_train is not None:
            datasets["训练集"] = (np.asarray(context.y_train), np.asarray(context.y_score_train))
        if context.y_test is not None and context.y_score_test is not None:
            datasets["测试集"] = (np.asarray(context.y_test), np.asarray(context.y_score_test))
        if context.y_oot is not None and context.y_score_oot is not None:
            datasets["跨时间验证集"] = (np.asarray(context.y_oot), np.asarray(context.y_score_oot))

        if datasets:
            effect_df = ModelEffectOperator.compute_effect_table(datasets, context.metrics)
            subs.append(SubSection(title="1.6模型效果汇总", data=effect_df))

        return ReportSectionResult(
            sheet_name=self.title,
            sub_sections=subs,
        )
