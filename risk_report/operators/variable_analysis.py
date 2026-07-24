"""VariableAnalysisOperator — 变量分析（2.1~2.4）。"""

import numpy as np
import pandas as pd

from .._base import ReportOperator, ReportSectionResult, SubSection
from .._context import ReportContext
from .._scoring import compute_per_feature_ks


class VariableAnalysisOperator(ReportOperator):
    """变量分析算子 — 产出 2.1~2.4 四个子章节。"""

    @property
    def name(self) -> str:
        return "variable_analysis"

    @property
    def title(self) -> str:
        return "2.变量分析"

    def compute(self, context: ReportContext) -> ReportSectionResult:
        subs = []
        attrs = context.pipeline_attrs

        # 2.1 变量描述
        subs.append(SubSection(
            title="1.变量描述",
            data=pd.DataFrame([{"说明": "经过变量初筛后的特征描述，详见附件3-变量描述"}]),
        ))

        # 2.2 变量清洗
        cleaning_rows = []
        if attrs and attrs.drop_columns_:
            cleaning_rows.append({"清洗步骤": "删除列", "数量": len(attrs.drop_columns_), "列名": ", ".join(attrs.drop_columns_[:10]) + ("..." if len(attrs.drop_columns_) > 10 else "")})
        if attrs and attrs.impute_values_:
            cleaning_rows.append({"清洗步骤": "缺失填充", "数量": len(attrs.impute_values_), "列名": ""})
        if cleaning_rows:
            subs.append(SubSection(
                title="2.变量清洗",
                data=pd.DataFrame(cleaning_rows),
                note="撰写变量清洗逻辑，包括缺失值、异常值清洗方案",
            ))
        else:
            subs.append(SubSection(
                title="2.变量清洗",
                data=pd.DataFrame([{"说明": "变量清洗逻辑需手动补充"}]),
            ))

        # 2.3 变量筛选（委托 FeatureFilterSummaryOperator）
        from .feature_filter import FeatureFilterSummaryOperator
        filter_result = FeatureFilterSummaryOperator().compute(context)
        if filter_result.sub_sections:
            subs.append(filter_result.sub_sections[0])

        # 2.4 变量分析表
        analysis_rows = []
        if attrs and attrs.feature_names_in_:
            # 获取各指标数据
            iv_series = attrs.iv_values_ if attrs.iv_values_ is not None else pd.Series(dtype=float)
            psi_series = attrs.psi_values_ if attrs.psi_values_ is not None else pd.Series(dtype=float)
            gain_dict = attrs.feature_importance_gain_ or {}
            weight_dict = attrs.feature_importance_weight_ or {}
            gain_total = attrs.gain_total_ or 1
            weight_total = attrs.weight_total_ or 1

            # 缺失率
            missing_train = pd.Series(dtype=float)
            missing_oot = pd.Series(dtype=float)
            if context.X_train is not None:
                missing_train = context.X_train.isnull().mean()
            if context.X_oot is not None:
                missing_oot = context.X_oot.isnull().mean()

            # 单特征 KS
            ks_train = pd.Series(dtype=float)
            ks_oot = pd.Series(dtype=float)
            if context.X_train is not None and context.y_score_train is not None:
                ks_train = compute_per_feature_ks(
                    context.X_train[attrs.feature_names_in_],
                    context.y_train, context.y_score_train,
                )
            if context.X_oot is not None and context.y_score_oot is not None:
                ks_oot = compute_per_feature_ks(
                    context.X_oot[attrs.feature_names_in_],
                    context.y_oot, context.y_score_oot,
                )

            for col in attrs.feature_names_in_:
                meta = context.feature_meta.get(col, {}) if context.feature_meta else {}
                analysis_rows.append({
                    "feature": col,
                    "变量含义": meta.get("含义", "未提供"),
                    "来源": meta.get("来源", "未提供"),
                    "类别": meta.get("类别", "未提供"),
                    "缺失率_train": missing_train.get(col, 0),
                    "缺失率_oot": missing_oot.get(col, 0),
                    "iv_train": iv_series.get(col, 0),
                    "iv_oot": iv_series.get(col, 0),
                    "ks_train": ks_train.get(col, 0),
                    "ks_oot": ks_oot.get(col, 0),
                    "gain": gain_dict.get(col, 0),
                    "gain_per": gain_dict.get(col, 0) / gain_total,
                    "weight": weight_dict.get(col, 0),
                    "weight_per": weight_dict.get(col, 0) / weight_total,
                    "psi": psi_series.get(col, 0),
                })

        if analysis_rows:
            subs.append(SubSection(
                title="4.变量分析",
                data=pd.DataFrame(analysis_rows),
            ))
        else:
            subs.append(SubSection(
                title="4.变量分析",
                data=pd.DataFrame([{"说明": "无入模特征数据"}]),
            ))

        return ReportSectionResult(
            sheet_name=self.title,
            sub_sections=subs,
        )
