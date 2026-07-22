"""
分箱基类 — BaseBinner

定义分箱算子的通用接口，所有分箱策略都应继承此类。
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .._base import RiskTransformer, validate_dataframe


class BaseBinner(RiskTransformer, ABC):
    """
    分箱算子抽象基类。

    子类必须实现 _bin_column(x, y) 方法来定义具体的分箱策略。
    基类负责遍历所有列并收集分箱结果。

    Attributes
    ----------
    bin_edges_ : dict
        {col_name: np.ndarray} 各列的分箱边界点。
    bin_labels_ : dict
        {col_name: list[str]} 各列的箱标签。
    """

    def fit(self, X, y=None):
        """
        对每列计算分箱边界。

        Args:
            X: pandas DataFrame
            y: 目标变量（分箱需要监督信息）

        Returns:
            self
        """
        X = validate_dataframe(X)
        if y is None:
            raise ValueError("分箱算子需要目标变量 y，不能为 None")
        y = pd.Series(y, index=X.index)

        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]

        self.bin_edges_ = {}
        self.bin_labels_ = {}

        for col in X.columns:
            x_col = X[col].values
            edges, labels = self._bin_column(x_col, y.values)
            self.bin_edges_[col] = edges
            self.bin_labels_[col] = labels

        return self

    def transform(self, X):
        """
        将连续值映射为箱索引（0-based 整数）。
        分类列使用映射字典转换。

        Args:
            X: pandas DataFrame

        Returns:
            DataFrame，每列值为整数箱索引
        """
        validate_dataframe(X)
        X_out = pd.DataFrame(index=X.index)

        for col in X.columns:
            if col not in self.bin_edges_:
                raise ValueError(
                    f"列 '{col}' 未在 fit 时训练，无法分箱"
                )
            edges = self.bin_edges_[col]

            # 分类列：使用映射而非 pd.cut
            if hasattr(self, '_categorical_cols_') and col in self._categorical_cols_:
                # 分类列的 edges 是整数映射边界，需要先把原始值映射为整数
                cat_map = self._cat_maps_.get(col, {})
                x_mapped = X[col].map(cat_map)
                X_out[col] = pd.cut(
                    pd.to_numeric(x_mapped, errors="coerce"),
                    bins=edges,
                    labels=False,
                    include_lowest=True,
                )
            else:
                # 连续列：强制数值型，再 pd.cut
                x_numeric = pd.to_numeric(X[col], errors="coerce")
                X_out[col] = pd.cut(
                    x_numeric,
                    bins=edges,
                    labels=False,
                    include_lowest=True,
                )

        return X_out

    @abstractmethod
    def _bin_column(self, x, y):
        """
        对单列计算分箱边界。

        Args:
            x: np.ndarray，特征值
            y: np.ndarray，目标值

        Returns:
            edges: np.ndarray，分箱边界点
            labels: list[str]，箱标签
        """
        pass

    def get_bin_table(self, feature):
        """
        获取指定特征的分箱汇总表。

        Args:
            feature: 特征名

        Returns:
            DataFrame，包含 bin_index, bin_range, count 等列
        """
        if feature not in self.bin_edges_:
            raise ValueError(f"特征 '{feature}' 未训练")
        edges = self.bin_edges_[feature]
        labels = self.bin_labels_[feature]

        rows = []
        for i, label in enumerate(labels):
            rows.append({
                "bin_index": i,
                "bin_label": label,
                "bin_lower": edges[i],
                "bin_upper": edges[i + 1] if i + 1 < len(edges) else np.inf,
            })
        return pd.DataFrame(rows)
