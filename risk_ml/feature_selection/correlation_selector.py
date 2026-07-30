"""
相关性筛选算子 — CorrelationSelector

移除高度相关的冗余特征，保留 IV 较高的特征。
"""

import numpy as np
import pandas as pd

from .._base import RiskSelector, validate_dataframe


class CorrelationSelector(RiskSelector):
    """
    相关性筛选算子：移除高相关特征对中 IV 较低者。

    Parameters
    ----------
    corr_threshold : float, default=0.7
        Pearson 相关系数绝对值阈值，超过此值的特征对为高相关。
    iv_values : dict or pd.Series or None, default=None
        预计算的 IV 值，用于高相关特征对的保留决策。
        若为 None，使用方差作为替代指标。
    strategy : str, default='drop_one'
        高相关对的处理策略。
        'drop_one' — 保留 IV 较高者，删除 IV 较低者
        'drop_both' — 两个都删除
    max_samples : int, default=10000
        计算相关矩阵时的最大样本量，超过时随机采样。
        10000 行的 Pearson 标准误 ≈ 0.01，足以可靠识别 >0.7 的高相关。
        设为 0 或 None 禁用采样。

    Attributes
    ----------
    correlation_matrix_ : pd.DataFrame
        特征间相关矩阵。
    drop_features_ : list[str]
        被删除的特征列表。
    """

    def __init__(self, corr_threshold=0.7, iv_values=None, strategy="drop_one",
                 max_samples=10000):
        self.corr_threshold = corr_threshold
        self.iv_values = iv_values
        self.strategy = strategy
        self.max_samples = max_samples

    def fit(self, X, y=None):
        """
        计算相关矩阵并识别需删除的特征。

        Args:
            X: pandas DataFrame
            y: 忽略

        Returns:
            self
        """
        from risk_report._scoring import compute_correlation_matrix

        X = validate_dataframe(X)
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]

        # 采样（后续所有计算都基于采样数据）
        if self.max_samples and len(X) > self.max_samples:
            X_compute = X.sample(n=self.max_samples, random_state=42)
        else:
            X_compute = X

        # 计算相关矩阵（numpy corrcoef，比 pandas corr 快 100x+）
        corr_arr = np.corrcoef(X_compute.values, rowvar=False)
        self.correlation_matrix_ = pd.DataFrame(corr_arr, index=X.columns, columns=X.columns)

        # 构建 IV 优先级（高 IV → 高优先级 → 保留）
        if self.iv_values is not None:
            if isinstance(self.iv_values, dict):
                priority = pd.Series(self.iv_values)
            else:
                priority = self.iv_values
        else:
            # 使用方差作为替代（基于采样数据，避免全量计算）
            priority = X_compute.var()

        # 向量化识别高相关对并决定删除
        drop_features = self._select_drops(priority)

        self.drop_features_ = sorted(drop_features)
        return self

    def _select_drops(self, priority: pd.Series) -> set[str]:
        """向量化高相关对筛选（比逐对循环快 10x+）。"""
        corr_arr = np.abs(self.correlation_matrix_.values)
        n = len(self.correlation_matrix_.columns)
        cols = self.correlation_matrix_.columns.tolist()
        threshold = self.corr_threshold

        # 上三角 mask（避免重复和对角线）
        triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)

        # 一次性找到所有超阈值位置
        high_corr = (corr_arr > threshold) & triu_mask
        rows, cols_idx = np.where(high_corr)

        # 按相关系数绝对值降序排列（强相关优先处理）
        order = np.argsort(-corr_arr[rows, cols_idx])
        rows = rows[order]
        cols_idx = cols_idx[order]

        drop_features = set()
        for r, c in zip(rows, cols_idx):
            col_a, col_b = cols[r], cols[c]
            if col_a in drop_features or col_b in drop_features:
                continue

            if self.strategy == "drop_both":
                drop_features.add(col_a)
                drop_features.add(col_b)
            else:  # drop_one
                iv_a = priority.get(col_a, 0)
                iv_b = priority.get(col_b, 0)
                drop_col = col_b if iv_a >= iv_b else col_a
                drop_features.add(drop_col)

        return drop_features

    @staticmethod
    def compute_correlation_matrix(X, max_samples=10000):
        """静态方法: 直接调用，无需构造算子。

        Parameters
        ----------
        X : pd.DataFrame
            特征数据
        max_samples : int
            最大计算样本量

        Returns
        -------
        pd.DataFrame
            相关矩阵
        """
        from risk_report._scoring import compute_correlation_matrix
        return compute_correlation_matrix(X, max_samples=max_samples)

    @staticmethod
    def compute_high_corr_pairs(X, threshold=0.7, max_samples=10000):
        """静态方法: 识别高相关特征对，无需构造算子。

        Parameters
        ----------
        X : pd.DataFrame
            特征数据
        threshold : float
            相关系数阈值
        max_samples : int
            最大计算样本量

        Returns
        -------
        list[tuple[str, str, float]]
            高相关特征对列表
        """
        from risk_report._scoring import compute_high_corr_pairs
        return compute_high_corr_pairs(X, threshold=threshold, max_samples=max_samples)

    def _get_support_mask(self):
        """返回特征保留掩码：非删除列 = True"""
        mask = np.array([col not in self.drop_features_ for col in self.feature_names_in_])
        return mask
