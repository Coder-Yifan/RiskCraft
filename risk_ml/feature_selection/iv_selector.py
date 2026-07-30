"""
IV 值筛选算子 — IVSelector

基于信息值（Information Value）筛选特征。
IV 是风控领域最常用的特征预测能力指标。

IV 参考阈值：
    < 0.02  : 无预测能力
    0.02-0.1: 弱预测能力
    0.1-0.3 : 中等预测能力
    > 0.3   : 强预测能力（注意可能存在数据泄露）
"""

import numpy as np
import pandas as pd

from .._base import RiskSelector, validate_dataframe


class IVSelector(RiskSelector):
    """
    IV 值筛选算子：基于信息值阈值过滤特征。

    Parameters
    ----------
    iv_threshold : float, default=0.02
        最低 IV 阈值，低于此值的特征被剔除。
    max_iv : float, default=0.5
        最高 IV 阈值，高于此值的特征疑似数据泄露，被剔除。
    eps : float, default=0.001
        WOE 计算的平滑因子。

    Attributes
    ----------
    iv_values_ : pd.Series
        各特征的 IV 值，索引为特征名。
    """

    def __init__(self, iv_threshold=0.02, max_iv=0.5, eps=0.001):
        self.iv_threshold = iv_threshold
        self.max_iv = max_iv
        self.eps = eps

    def fit(self, X, y=None):
        """
        计算各特征的 IV 值。

        Args:
            X: pandas DataFrame（建议已 WOE 编码，但也可接受分箱数据）
            y: 目标变量（二分类 0/1）

        Returns:
            self
        """
        from risk_report._scoring import compute_iv_from_data

        X = validate_dataframe(X)
        if y is None:
            raise ValueError("IVSelector 需要目标变量 y")

        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]

        self.iv_values_ = compute_iv_from_data(X, y, eps=self.eps)
        return self

    def _get_support_mask(self):
        """
        返回特征保留掩码。

        保留条件：iv_threshold <= IV <= max_iv
        """
        mask = (self.iv_values_ >= self.iv_threshold) & (self.iv_values_ <= self.max_iv)
        return mask.values
