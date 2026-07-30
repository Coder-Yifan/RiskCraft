"""VarBinning 算子 — 变量分箱（按GAIN）。

每个入模特征一个 SubSection（分箱表）。
"""

import numpy as np
import pandas as pd

from .._base import ReportOperator, SubSection, placeholder_df
from .._scoring import compute_lift_table


class VarBinningOperator(ReportOperator):
    """变量分箱 — 每个入模特征一个 SubSection（分箱明细表）。"""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    @property
    def name(self) -> str:
        return "var_binning"

    @property
    def title(self) -> str:
        return "变量分箱（按GAIN）"

    def compute(self, context) -> list[SubSection]:
        attrs = context.pipeline_attrs
        if attrs is None or not attrs.feature_names_in_:
            return [SubSection(self.title, placeholder_df(
                "流水线属性未提取，请通过 ReportContext.pipeline 传入"
            ))]

        datasets = context.get_datasets()
        if not datasets:
            return [SubSection(self.title, placeholder_df(
                "数据集未提供，请通过 ReportContext.data + tag_col + label_col + score_col 传入"
            ))]

        subs = []
        # 使用训练集数据
        cn_name = "训练集"
        if cn_name not in datasets:
            cn_name = list(datasets.keys())[0]

        y_true, y_score = datasets[cn_name]
        mask_tag = context.data[context.tag_col] == {"训练集": "train", "测试集": "test", "跨时间验证集": "oot"}[cn_name]
        subset = context.data[mask_tag]

        # WOE 分箱表
        woe_map = attrs.woe_map_ or {}
        bin_edges = attrs.bin_edges_ or {}
        feature_meta = context.feature_meta or {}

        for col in attrs.feature_names_in_:
            cn_title = feature_meta.get(col, {}).get("含义", col)

            # 如果有 woe_map 和 bin_edges，使用 WOE 分箱
            if col in woe_map and col in bin_edges:
                woe_dict = woe_map[col]
                edges = bin_edges[col]
                rows = []
                for i, (bin_label, woe_val) in enumerate(woe_dict.items()):
                    # 计算每箱的好坏
                    if col in subset.columns:
                        if i < len(edges) - 1:
                            bin_mask = (subset[col] >= edges[i]) & (subset[col] < edges[i + 1])
                        else:
                            bin_mask = subset[col] >= edges[i]
                        bin_goods = int((subset.loc[bin_mask, context.label_col] == 0).sum()) if context.label_col in subset.columns else 0
                        bin_bads = int((subset.loc[bin_mask, context.label_col] == 1).sum()) if context.label_col in subset.columns else 0
                        bin_total = bin_goods + bin_bads
                    else:
                        bin_goods = bin_bads = bin_total = 0

                    rows.append({
                        "分箱": bin_label,
                        "WOE": round(woe_val, 4),
                        "goods": bin_goods,
                        "bads": bin_bads,
                        "total": bin_total,
                    })
                subs.append(SubSection(title=cn_title, data=pd.DataFrame(rows)))
            else:
                # 无 WOE 分箱数据，用等频分箱
                if col in subset.columns:
                    col_vals = subset[col].values
                    # 简单占位
                    subs.append(SubSection(
                        title=cn_title,
                        data=placeholder_df(f"变量 {col} 的分箱数据未提取"),
                    ))
                else:
                    subs.append(SubSection(
                        title=cn_title,
                        data=placeholder_df(f"变量 {col} 不在数据集中"),
                    ))

        return subs
