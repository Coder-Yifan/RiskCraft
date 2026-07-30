"""VarAnalysis 算子 — 4.变量分析。"""

import numpy as np
import pandas as pd

from .._base import ReportOperator, SubSection, placeholder_df
from .._scoring import compute_per_feature_ks


class VarAnalysisOperator(ReportOperator):
    """变量分析 — 特征明细表（iv/ks/gain/weight/psi/缺失率）。"""

    @property
    def name(self) -> str:
        return "var_analysis"

    @property
    def title(self) -> str:
        return "4.变量分析"

    def compute(self, context) -> list[SubSection]:
        attrs = context.pipeline_attrs
        if attrs is None or not attrs.feature_names_in_:
            return [SubSection(self.title, placeholder_df(
                "流水线属性未提取，请通过 ReportContext.pipeline 传入"
            ))]

        iv_series = attrs.iv_values_ if attrs.iv_values_ is not None else pd.Series(dtype=float)
        psi_series = attrs.psi_values_ if attrs.psi_values_ is not None else pd.Series(dtype=float)
        gain_dict = attrs.feature_importance_gain_ or {}
        weight_dict = attrs.feature_importance_weight_ or {}
        gain_total = attrs.gain_total_ or 1
        weight_total = attrs.weight_total_ or 1

        # 缺失率（从 data 计算）
        missing_train = pd.Series(dtype=float)
        missing_oot = pd.Series(dtype=float)
        if context.data is not None:
            for tag_val, series_target in [("train", missing_train), ("oot", missing_oot)]:
                mask = context.data[context.tag_col] == tag_val
                subset = context.data[mask]
                if len(subset) > 0:
                    for col in attrs.feature_names_in_:
                        if col in subset.columns:
                            series_target[col] = subset[col].isnull().mean()
            missing_train = missing_train
            missing_oot = missing_oot

        # 单特征 KS
        ks_train = pd.Series(dtype=float)
        ks_oot = pd.Series(dtype=float)
        datasets = context.get_datasets()
        if context.data is not None and context.score_col in context.data.columns:
            for tag_val, series_target, cn_name in [("train", ks_train, "训练集"), ("oot", ks_oot, "跨时间验证集")]:
                if cn_name in datasets:
                    y_true, y_score = datasets[cn_name]
                    mask_tag = context.data[context.tag_col] == tag_val
                    subset = context.data[mask_tag]
                    feat_cols = [c for c in attrs.feature_names_in_ if c in subset.columns]
                    if feat_cols:
                        series_target = compute_per_feature_ks(
                            subset[feat_cols], y_true, y_score,
                        )
                        if tag_val == "train":
                            ks_train = series_target
                        else:
                            ks_oot = series_target

        rows = []
        for col in attrs.feature_names_in_:
            meta = context.feature_meta.get(col, {}) if context.feature_meta else {}
            rows.append({
                "feature": col,
                "变量含义": meta.get("含义", "未提供"),
                "来源": meta.get("来源", "未提供"),
                "类别": meta.get("类别", "未提供"),
                "缺失率_train": missing_train.get(col, 0),
                "缺失率_oot": missing_oot.get(col, 0),
                "iv_train": iv_series.get(col, 0),
                "ks_train": ks_train.get(col, 0),
                "ks_oot": ks_oot.get(col, 0),
                "gain": gain_dict.get(col, 0),
                "gain_per": gain_dict.get(col, 0) / gain_total,
                "weight": weight_dict.get(col, 0),
                "weight_per": weight_dict.get(col, 0) / weight_total,
                "psi": psi_series.get(col, 0),
            })

        return [SubSection(self.title, pd.DataFrame(rows))]
