"""
PSI 稳定性筛选算子 — PSISelector

PSI (Population Stability Index) 衡量两个分布的差异程度，
常用于监控特征分布的时序漂移。

PSI 参考阈值：
    < 0.1  : 稳定
    0.1-0.25: 边际（需关注）
    > 0.25 : 不稳定（应删除或重新训练）
"""

import numpy as np
import pandas as pd

from .._base import RiskSelector, validate_dataframe


class PSISelector(RiskSelector):
    """
    PSI 稳定性筛选算子：基于群体稳定性指数过滤特征。

    Parameters
    ----------
    psi_threshold : float, default=0.25
        PSI 阈值，高于此值的特征被标记为不稳定并移除。
    n_bins : int, default=10
        PSI 计算时的分箱数。
    eps : float, default=1e-4
        平滑因子，防止零概率箱。

    Attributes
    ----------
    reference_dist_ : dict
        {col: np.ndarray} 参考分布（fit 时记录）。
    psi_values_ : pd.Series
        各特征的 PSI 值（transform 时计算）。
    """

    def __init__(self, psi_threshold=0.25, n_bins=10, eps=1e-4):
        self.psi_threshold = psi_threshold
        self.n_bins = n_bins
        self.eps = eps

    def fit(self, X, y=None):
        """
        记录参考分布（训练集的分箱比例）。

        Args:
            X: pandas DataFrame（参考数据集，通常为训练集）
            y: 忽略

        Returns:
            self
        """
        X = validate_dataframe(X)
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]

        self.reference_dist_ = {}
        self._bin_edges_ = {}  # 保存分箱边界，transform 时复用

        for col in X.columns:
            col_data = X[col].dropna()
            if len(col_data) == 0:
                self.reference_dist_[col] = np.ones(self.n_bins) / self.n_bins
                self._bin_edges_[col] = None
                continue

            # 按分位数分箱，计算参考分布
            try:
                bins = np.percentile(
                    col_data, np.linspace(0, 100, self.n_bins + 1)
                )
                bins = np.unique(bins)
                if len(bins) <= 1:
                    self.reference_dist_[col] = np.array([1.0])
                    self._bin_edges_[col] = None
                    continue
                counts, _ = np.histogram(col_data, bins=bins)
                proportions = (counts + self.eps) / (counts.sum() + self.eps * len(counts))
                self.reference_dist_[col] = proportions
                self._bin_edges_[col] = bins
            except Exception:
                self.reference_dist_[col] = np.ones(self.n_bins) / self.n_bins
                self._bin_edges_[col] = None

        # 初始化 psi_values_（fit 时不计算，transform 时才比较）
        self.psi_values_ = pd.Series(0.0, index=X.columns)
        return self

    def transform(self, X):
        """
        计算各特征 PSI 并移除不稳定特征。

        Args:
            X: pandas DataFrame（待评估数据集）

        Returns:
            筛选后的 DataFrame
        """
        validate_dataframe(X)

        # 计算 PSI
        psi_dict = {}
        for col in X.columns:
            if col not in self.reference_dist_:
                raise ValueError(f"特征 '{col}' 未在 fit 时训练")

            ref_dist = self.reference_dist_[col]
            col_data = X[col].dropna()

            if len(col_data) == 0:
                psi_dict[col] = 0.0
                continue

            try:
                # 使用 fit 时保存的分箱边界（确保两组数据用同一标准对比）
                saved_edges = self._bin_edges_.get(col)
                if saved_edges is None:
                    psi_dict[col] = 0.0
                    continue

                # 当前分布在同一分箱下的计数
                cur_counts, _ = np.histogram(col_data, bins=saved_edges)
                cur_dist = (cur_counts + self.eps) / (cur_counts.sum() + self.eps * len(cur_counts))

                # 对齐长度
                min_len = min(len(ref_dist), len(cur_dist))
                psi = 0.0
                for i in range(min_len):
                    p = ref_dist[i]
                    q = cur_dist[i]
                    if p > 0 and q > 0:
                        psi += (q - p) * np.log(q / p)

                psi_dict[col] = max(0.0, psi)
            except Exception:
                psi_dict[col] = 0.0

        self.psi_values_ = pd.Series(psi_dict, index=X.columns)

        # 使用 SelectorMixin 的 transform 进行筛选
        return super().transform(X)

    def _get_support_mask(self):
        """返回特征保留掩码：PSI <= threshold"""
        if not hasattr(self, "psi_values_") or self.psi_values_ is None:
            return np.ones(self.n_features_in_, dtype=bool)
        return (self.psi_values_ <= self.psi_threshold).values
