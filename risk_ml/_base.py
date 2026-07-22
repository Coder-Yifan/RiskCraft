"""
risk_ml 基类模块

提供所有风控建模算子的公共基类，确保 sklearn 兼容性：
- RiskTransformer: 转换器基类（fit/transform），默认 pandas 输出
- RiskSelector:    特征筛选器基类（fit/transform + _get_support_mask）
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted


def validate_dataframe(X, reset=False):
    """
    校验输入是否为 pandas DataFrame，并记录特征元信息。

    Args:
        X: 输入数据，必须为 pandas DataFrame
        reset: 是否重置特征元信息（首次 fit 时为 True）

    Returns:
        校验后的 DataFrame

    Raises:
        TypeError: 输入不是 DataFrame
        ValueError: DataFrame 为空
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"risk_ml 算子要求输入为 pandas.DataFrame，"
            f"收到: {type(X).__name__}"
        )
    if X.shape[1] == 0:
        raise ValueError("输入 DataFrame 没有列（0 个特征）")
    return X


class RiskTransformer(BaseEstimator, TransformerMixin):
    """
    风控转换器基类。

    所有 risk_ml 的 Transformer 都应继承此类。
    自动提供：
    - fit_transform()（来自 TransformerMixin）
    - get_params() / set_params()（来自 BaseEstimator）
    - 默认 pandas 输出（set_output(transform="pandas")）

    子类必须实现：
    - fit(self, X, y=None) → self
    - transform(self, X) → DataFrame
    """

    def fit(self, X, y=None):
        """
        拟合转换器。子类必须重写此方法。

        Args:
            X: pandas DataFrame
            y: 目标变量（可选）

        Returns:
            self
        """
        X = validate_dataframe(X)
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """
        转换数据。子类必须重写此方法。

        Args:
            X: pandas DataFrame

        Returns:
            转换后的 DataFrame
        """
        raise NotImplementedError("子类必须实现 transform()")

    def get_feature_names_out(self, input_features=None):
        """
        返回转换后的特征名（sklearn set_output 需要）。
        """
        check_is_fitted(self, "feature_names_in_")
        return np.array(self.feature_names_in_)

    def _more_tags(self):
        """sklearn 标签：允许 NaN 输入（风控数据常见缺失值）"""
        return {"allow_nan": True}


class RiskSelector(BaseEstimator, SelectorMixin):
    """
    风控特征筛选器基类。

    所有 risk_ml 的 Selector 都应继承此类。
    自动提供：
    - transform()（来自 SelectorMixin，基于 _get_support_mask 过滤列）
    - get_support()（来自 SelectorMixin）
    - inverse_transform()（来自 SelectorMixin）
    - get_feature_names_out()（来自 SelectorMixin）
    - get_params() / set_params()（来自 BaseEstimator）

    子类必须实现：
    - fit(self, X, y=None) → self
    - _get_support_mask(self) → np.ndarray（布尔数组）
    """

    def fit(self, X, y=None):
        """
        拟合筛选器。子类必须重写此方法。

        Args:
            X: pandas DataFrame
            y: 目标变量（可选）

        Returns:
            self
        """
        X = validate_dataframe(X)
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]
        return self

    @abstractmethod
    def _get_support_mask(self):
        """
        返回特征保留掩码。

        Returns:
            np.ndarray: 形状 (n_features_in_,)，True 表示保留该特征
        """
        pass

    def _more_tags(self):
        """sklearn 标签：允许 NaN 输入"""
        return {"allow_nan": True}

    def transform(self, X):
        """
        覆盖 SelectorMixin.transform，确保返回 DataFrame。

        SelectorMixin 默认返回 ndarray，但风控场景需要保留列名。
        """
        X = validate_dataframe(X)
        mask = self._get_support_mask()
        keep_cols = [c for c, m in zip(self.feature_names_in_, mask) if m]
        return X[keep_cols]

    def get_feature_names_out(self, input_features=None):
        """
        返回筛选后的特征名（sklearn set_output 需要）。
        """
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "feature_names_in_")
        mask = self._get_support_mask()
        return np.array([c for c, m in zip(self.feature_names_in_, mask) if m])
