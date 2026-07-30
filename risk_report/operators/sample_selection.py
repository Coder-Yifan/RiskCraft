"""SampleSelection 算子 — 1.4样本选择。

原始样本分布: 优先使用外部传入的 sample_origin_distribution，
  无外部数据时自动从 context.data 计算简化版。
开发样本分布: 自动从 context.data 计算（特征筛选只删特征不删样本）。
"""

import pandas as pd

from .._base import ReportOperator, SubSection, placeholder_df


class SampleSelectionOperator(ReportOperator):
    """样本选择 — 原始样本分布表 + 开发样本分布表。

    原始样本分布:
    - 优先使用 context.sample_origin_distribution（外部传入，含筛选前全量样本）
    - 无外部数据时，自动从 context.data 计算简化版（仅建模样本）

    开发样本分布:
    - 自动从 context.data 计算（train/test/oot 好坏灰统计）
    """

    @property
    def name(self) -> str:
        return "sample_selection"

    @property
    def title(self) -> str:
        return "1.4样本选择"

    def compute(self, context) -> list[SubSection]:
        subs = []

        # 1. 原始样本分布
        if context.sample_origin_distribution is not None:
            subs.append(SubSection("原始样本分布", context.sample_origin_distribution))
        else:
            df_origin = self._compute_origin_distribution(context)
            if df_origin is not None:
                subs.append(SubSection(
                    "原始样本分布",
                    df_origin,
                    note="以下为建模样本分布，不含筛选前被排除的样本；完整原始样本分布请通过 ReportContext.sample_origin_distribution 传入",
                ))
            else:
                subs.append(SubSection("原始样本分布", placeholder_df(
                    "原始样本分布数据未提供，请通过 ReportContext.sample_origin_distribution 传入"
                ), note="数据需外部提供"))

        # 2. 开发样本分布（自动计算）
        df_dev = self._compute_dev_distribution(context)
        if df_dev is not None:
            subs.append(SubSection("开发样本分布", df_dev))
        else:
            subs.append(SubSection("开发样本分布", placeholder_df(
                "数据集未提供，请通过 ReportContext.data + tag_col + label_col 传入"
            )))

        return subs

    @staticmethod
    def _compute_origin_distribution(context) -> pd.DataFrame | None:
        """从 context.data 计算简化版原始样本分布。"""
        if context.data is None or context.label_col is None:
            return None
        if context.label_col not in context.data.columns:
            return None

        from .._context import TAG_CN_MAP
        rows = []
        for tag_val, cn_name in TAG_CN_MAP.items():
            mask = context.data[context.tag_col] == tag_val
            subset = context.data[mask]
            if len(subset) == 0:
                continue
            label = subset[context.label_col]
            goods = int((label == 0).sum())
            bads = int((label == 1).sum())
            total = goods + bads
            rows.append({
                "数据集": cn_name,
                "样本量": total,
                "好样本": goods,
                "坏样本": bads,
                "坏占比": bads / total if total > 0 else 0,
            })

        # 总计
        if rows:
            total_all = sum(r["样本量"] for r in rows)
            goods_all = sum(r["好样本"] for r in rows)
            bads_all = sum(r["坏样本"] for r in rows)
            rows.append({
                "数据集": "总计",
                "样本量": total_all,
                "好样本": goods_all,
                "坏样本": bads_all,
                "坏占比": bads_all / total_all if total_all > 0 else 0,
            })

        return pd.DataFrame(rows) if rows else None

    @staticmethod
    def _compute_dev_distribution(context) -> pd.DataFrame | None:
        """从 context.data 自动计算开发样本分布（train/test/oot 好坏灰统计）。"""
        stats = context.get_sample_stats()
        if not stats:
            return None

        rows = []
        total_goods = 0
        total_bads = 0
        total_gray = 0
        total_n = 0

        for cn_name, s in stats.items():
            rows.append({
                "数据集": cn_name,
                "好样本": s["goods"],
                "坏样本": s["bads"],
                "灰样本": s["gray"],
                "总量": s["total"],
                "好占比": s["goods"] / s["total"] if s["total"] > 0 else 0,
                "坏占比": s["bad_rate"],
                "灰占比": s["gray"] / s["total"] if s["total"] > 0 else 0,
            })
            total_goods += s["goods"]
            total_bads += s["bads"]
            total_gray += s["gray"]
            total_n += s["total"]

        # 总计
        if rows:
            rows.append({
                "数据集": "总计",
                "好样本": total_goods,
                "坏样本": total_bads,
                "灰样本": total_gray,
                "总量": total_n,
                "好占比": total_goods / total_n if total_n > 0 else 0,
                "坏占比": total_bads / total_n if total_n > 0 else 0,
                "灰占比": total_gray / total_n if total_n > 0 else 0,
            })

        return pd.DataFrame(rows) if rows else None
