"""VarFilter 算子 — 3.变量筛选。"""

import pandas as pd

from .._base import ReportOperator, SubSection, placeholder_df


class VarFilterOperator(ReportOperator):
    """变量筛选 — 筛选步骤表（筛选指标/阈值/筛选数量/剩余数量）。"""

    @property
    def name(self) -> str:
        return "var_filter"

    @property
    def title(self) -> str:
        return "3.变量筛选"

    def compute(self, context) -> list[SubSection]:
        attrs = context.pipeline_attrs
        if attrs is None:
            return [SubSection(self.title, placeholder_df(
                "流水线属性未提取，请通过 ReportContext.pipeline 传入"
            ))]

        # 计算初始特征数
        initial_n = None
        if attrs.feature_names_in_ is not None:
            final_n = len(attrs.feature_names_in_)
        else:
            final_n = 0

        # 从 data 推算初始数
        if context.data is not None:
            feature_cols = context._get_feature_columns()
            initial_n = len(feature_cols)
        elif attrs.n_features_in_ is not None:
            initial_n = attrs.n_features_in_

        rows = []
        remaining = initial_n

        # 缺失率筛选
        threshold = attrs.missing_threshold_
        drop_count = len(attrs.drop_columns_) if attrs.drop_columns_ else 0
        rows.append({
            "筛选指标": "缺失率",
            "阈值": f"＜{threshold}" if threshold else "未配置",
            "筛选数量": drop_count,
            "剩余数量": remaining - drop_count if remaining else "未知",
        })
        if remaining:
            remaining -= drop_count

        # IV 筛选
        if attrs.iv_values_ is not None and attrs.iv_threshold_ is not None:
            iv_filter_count = int(((attrs.iv_values_ < attrs.iv_threshold_) | (attrs.iv_values_ > (attrs.iv_max_iv_ or 0.5))).sum())
            rows.append({
                "筛选指标": "IV",
                "阈值": f"[{attrs.iv_threshold_}, {attrs.iv_max_iv_ or 0.5}]",
                "筛选数量": iv_filter_count,
                "剩余数量": remaining - iv_filter_count if remaining else "未知",
            })
            if remaining:
                remaining -= iv_filter_count
        else:
            rows.append({"筛选指标": "IV", "阈值": "未配置", "筛选数量": "未知", "剩余数量": remaining or "未知"})

        # PSI 稳定性筛选
        if attrs.psi_values_ is not None and attrs.psi_threshold_ is not None:
            psi_filter_count = int((attrs.psi_values_ > attrs.psi_threshold_).sum())
            rows.append({
                "筛选指标": "PSI",
                "阈值": f"＜{attrs.psi_threshold_}",
                "筛选数量": psi_filter_count,
                "剩余数量": remaining - psi_filter_count if remaining else "未知",
            })
            if remaining:
                remaining -= psi_filter_count
        else:
            rows.append({"筛选指标": "PSI", "阈值": "未配置", "筛选数量": "未知", "剩余数量": remaining or "未知"})

        # 相关性筛选
        if attrs.drop_features_ is not None and attrs.corr_threshold_ is not None:
            corr_filter_count = len(attrs.drop_features_)
            rows.append({
                "筛选指标": "相关性",
                "阈值": f"＜{attrs.corr_threshold_}",
                "筛选数量": corr_filter_count,
                "剩余数量": remaining - corr_filter_count if remaining else "未知",
            })
            if remaining:
                remaining -= corr_filter_count
        else:
            rows.append({"筛选指标": "相关性", "阈值": "未配置", "筛选数量": "未知", "剩余数量": remaining or "未知"})

        # 重要性筛选
        rows.append({
            "筛选指标": "重要性",
            "阈值": "≠0",
            "筛选数量": 0,
            "剩余数量": final_n or remaining,
        })

        return [SubSection(self.title, pd.DataFrame(rows))]
