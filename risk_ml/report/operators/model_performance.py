"""ModelPerformanceOperator — 模型表现（3.1~3.4）。"""

import numpy as np
import pandas as pd

from .._base import ReportOperator, ReportSectionResult, SubSection
from .._context import ReportContext
from .._scoring import compute_lift_table, compute_sample_stats
from .model_effect import ModelEffectOperator
from .score_lift import ScoreLiftOperator


class ModelPerformanceOperator(ReportOperator):
    """模型表现算子 — 产出 3.1~3.4 四个子章节。"""

    @property
    def name(self) -> str:
        return "model_performance"

    @property
    def title(self) -> str:
        return "3.模型表现"

    def compute(self, context: ReportContext) -> ReportSectionResult:
        subs = []
        attrs = context.pipeline_attrs

        # 3.1 建模方法选择
        if attrs and attrs.model_params_:
            # XGB/LGBM 参数表
            params = attrs.model_params_
            # 过滤出关键参数
            key_params = {
                "max_depth": params.get("max_depth", ""),
                "n_estimators": params.get("n_estimators", ""),
                "learning_rate": params.get("learning_rate", params.get("eta", "")),
                "subsample": params.get("subsample", ""),
                "colsample_bytree": params.get("colsample_bytree", ""),
                "reg_alpha": params.get("reg_alpha", params.get("alpha", "")),
                "reg_lambda": params.get("reg_lambda", params.get("lambda", "")),
                "min_child_weight": params.get("min_child_weight", ""),
            }
            # 参数名中文映射
            desc_map = {
                "max_depth": "最大树深",
                "n_estimators": "决策树个数",
                "learning_rate": "学习速率",
                "subsample": "子树样本采样比例",
                "colsample_bytree": "子树样本训练特征采样比例",
                "reg_alpha": "L1正则化惩罚系数",
                "reg_lambda": "L2正则化惩罚系数",
                "min_child_weight": "叶子节点中最小权重值",
            }
            param_rows = []
            for name, val in key_params.items():
                if val != "":
                    param_rows.append({
                        "参数名称解释": desc_map.get(name, name),
                        "参数名称": name,
                        "数值": str(val),
                    })
            subs.append(SubSection(
                title="1.建模方法选择",
                data=pd.DataFrame(param_rows),
            ))
        else:
            subs.append(SubSection(
                title="1.建模方法选择",
                data=pd.DataFrame([{"说明": "模型参数未提取"}]),
            ))

        # 3.2 模型效果
        datasets = {}
        if context.y_train is not None and context.y_score_train is not None:
            datasets["训练集"] = (np.asarray(context.y_train), np.asarray(context.y_score_train))
        if context.y_test is not None and context.y_score_test is not None:
            datasets["测试集"] = (np.asarray(context.y_test), np.asarray(context.y_score_test))
        if context.y_oot is not None and context.y_score_oot is not None:
            datasets["跨时间验证集"] = (np.asarray(context.y_oot), np.asarray(context.y_score_oot))

        if datasets:
            effect_df = ModelEffectOperator.compute_effect_table(
                datasets, context.metrics, lift_percentiles=[10, 5, 2, 1],
            )
            subs.append(SubSection(title="2.模型效果", data=effect_df))

        # 3.3 模型分分箱表现（不含灰样本）
        lift_subs = []
        for ds_name, y, y_score in [
            ("TRAIN", context.y_train, context.y_score_train),
            ("TEST", context.y_test, context.y_score_test),
            ("OOT", context.y_oot, context.y_score_oot),
        ]:
            if y is not None and y_score is not None:
                y_true = np.asarray(y)
                y_score_arr = np.asarray(y_score)
                baseline = context.baseline_scores.get(ds_name.lower()) if context.baseline_scores else None
                df = compute_lift_table(y_true, y_score_arr, n_bins=10, baseline_score=baseline)
                lift_subs.append(SubSection(title=ds_name, data=df))

        if lift_subs:
            subs.append(SubSection(
                title="3.模型分分箱表现（不含灰样本）",
                data=pd.DataFrame([{"说明": "各数据集 lift 表详见子章节"}]),
                note="以下为各数据集的分箱表现",
            ))
            subs.extend(lift_subs)

        # 3.4 模型分分箱表现（含灰样本）— 仅在有灰样本时
        if context.X_gray is not None and context.y_gray is not None:
            gray_subs = []
            for ds_name, y, y_score in [
                ("TRAIN_GRAY", context.y_train, context.y_score_train),
            ]:
                if y is not None and y_score is not None:
                    # 合并灰样本到训练集
                    y_combined = np.concatenate([np.asarray(y), np.asarray(context.y_gray)])
                    y_score_combined = np.concatenate([np.asarray(y_score)])
                    baseline = None
                    df = compute_lift_table(y_combined, y_score_combined, n_bins=10, baseline_score=baseline)
                    gray_subs.append(SubSection(title="TRAIN(含灰)", data=df))

            if gray_subs:
                subs.append(SubSection(
                    title="4.模型分分箱表现（含灰样本）",
                    data=pd.DataFrame([{"说明": "含灰样本的分箱表现"}]),
                ))
                subs.extend(gray_subs)

        return ReportSectionResult(
            sheet_name=self.title,
            sub_sections=subs,
        )
