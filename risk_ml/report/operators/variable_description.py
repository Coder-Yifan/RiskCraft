"""VariableDescriptionOperator — 附件3-变量描述。"""

import numpy as np
import pandas as pd

from .._base import ReportOperator, ReportSectionResult, SubSection
from .._context import ReportContext


class VariableDescriptionOperator(ReportOperator):
    """变量描述算子 — 附件3（变量范围描述 + 分位数分布）。"""

    @property
    def name(self) -> str:
        return "variable_description"

    @property
    def title(self) -> str:
        return "附件3-变量描述"

    def compute(self, context: ReportContext) -> ReportSectionResult:
        attrs = context.pipeline_attrs
        rows = []

        # 确定要描述的特征列表
        features = attrs.feature_names_in_ if attrs and attrs.feature_names_in_ else []

        if not features and context.X_train is not None:
            features = list(context.X_train.columns)

        if context.X_train is not None and features:
            X = context.X_train[features]
            quantiles = X.quantile([0, 0.25, 0.5, 0.75, 1])

            for col in features:
                meta = context.feature_meta.get(col, {}) if context.feature_meta else {}
                q_vals = quantiles[col]
                rows.append({
                    "变量名": col,
                    "变量含义": meta.get("含义", "未提供"),
                    "来源": meta.get("来源", "未提供"),
                    "类别": meta.get("类别", "未提供"),
                    "最小值": q_vals[0],
                    "25%分位": q_vals[0.25],
                    "中位数": q_vals[0.5],
                    "75%分位": q_vals[0.75],
                    "最大值": q_vals[1],
                    "缺失率": X[col].isnull().mean(),
                    "唯一值占比": X[col].nunique() / len(X),
                })

        if rows:
            df = pd.DataFrame(rows)
        else:
            df = pd.DataFrame([{"说明": "无特征数据，请提供 X_train 或 pipeline_attrs.feature_names_in_"}])

        return ReportSectionResult(
            sheet_name=self.title,
            sub_sections=[SubSection(title="变量范围描述", data=df)],
        )
