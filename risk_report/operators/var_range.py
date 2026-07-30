"""VarRange 算子 — 变量范围描述。"""

import numpy as np
import pandas as pd

from .._base import ReportOperator, SubSection, placeholder_df


class VarRangeOperator(ReportOperator):
    """变量范围描述 — 特征范围表（五分位数/缺失率/异常值/上线可靠性）。"""

    @property
    def name(self) -> str:
        return "var_range"

    @property
    def title(self) -> str:
        return "变量范围描述"

    def compute(self, context) -> list[SubSection]:
        attrs = context.pipeline_attrs
        features = attrs.feature_names_in_ if attrs and attrs.feature_names_in_ else []

        if not features:
            return [SubSection(self.title, placeholder_df(
                "无入模特征数据，请通过 ReportContext.pipeline 传入"
            ))]

        if context.data is None:
            return [SubSection(self.title, placeholder_df(
                "数据集未提供，请通过 ReportContext.data 传入"
            ))]

        # 使用训练集数据计算
        mask = context.data[context.tag_col] == "train"
        subset = context.data[mask]
        if len(subset) == 0:
            # 兜底：使用全部数据
            subset = context.data

        feat_cols = [c for c in features if c in subset.columns]
        if not feat_cols:
            return [SubSection(self.title, placeholder_df(
                "入模特征不在数据集中"
            ))]

        X = subset[feat_cols]
        quantiles = X.quantile([0, 0.25, 0.5, 0.75, 1])

        rows = []
        for col in feat_cols:
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

        return [SubSection(self.title, pd.DataFrame(rows))]
