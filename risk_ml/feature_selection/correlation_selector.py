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

    Attributes
    ----------
    correlation_matrix_ : pd.DataFrame
        特征间相关矩阵。
    drop_features_ : list[str]
        被删除的特征列表。
    """

    def __init__(self, corr_threshold=0.7, iv_values=None, strategy="drop_one"):
        self.corr_threshold = corr_threshold
        self.iv_values = iv_values
        self.strategy = strategy

    def fit(self, X, y=None):
        """
        计算相关矩阵并识别需删除的特征。

        Args:
            X: pandas DataFrame
            y: 忽略

        Returns:
            self
        """
        X = validate_dataframe(X)
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]

        # 计算相关矩阵
        self.correlation_matrix_ = X.corr()

        # 构建 IV 优先级（高 IV → 高优先级 → 保留）
        if self.iv_values is not None:
            if isinstance(self.iv_values, dict):
                priority = pd.Series(self.iv_values)
            else:
                priority = self.iv_values
        else:
            # 使用方差作为替代
            priority = X.var()

        # 识别高相关对并决定删除
        drop_features = set()
        cols = X.columns.tolist()

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                col_a, col_b = cols[i], cols[j]
                if col_a in drop_features or col_b in drop_features:
                    continue

                corr_val = abs(self.correlation_matrix_.loc[col_a, col_b])
                if corr_val > self.corr_threshold:
                    if self.strategy == "drop_both":
                        drop_features.add(col_a)
                        drop_features.add(col_b)
                    else:  # drop_one
                        iv_a = priority.get(col_a, 0)
                        iv_b = priority.get(col_b, 0)
                        drop_col = col_b if iv_a >= iv_b else col_a
                        drop_features.add(drop_col)

        self.drop_features_ = sorted(drop_features)
        return self

    def _get_support_mask(self):
        """返回特征保留掩码：非删除列 = True"""
        mask = np.array([col not in self.drop_features_ for col in self.feature_names_in_])
        return mask
