"""
WOE 编码算子 — WoeEncoder / BinnerWoeEncoder

WOE (Weight of Evidence) 是风控领域核心编码方法：
    WOE(bin) = ln(dist_pos(bin) / dist_neg(bin))
    IV(feature) = Σ (dist_pos - dist_neg) × WOE

设计决策：WOE 与分箱解耦。
- WoeEncoder: 接受已分箱的 X（整数箱索引），计算 WOE 并替换
- BinnerWoeEncoder: 便捷联合算子，内部串联 ChiMergeBinner → WoeEncoder
"""

import numpy as np
import pandas as pd

from .._base import RiskTransformer, validate_dataframe
from ..binning import ChiMergeBinner


class WoeEncoder(RiskTransformer):
    """
    WOE 编码算子：将分箱结果转换为 WOE 值。

    Parameters
    ----------
    binner : ChiMergeBinner or None, default=None
        如提供，fit 时先调用 binner.transform(X) 进行分箱。
        如为 None，输入 X 须已是整数箱索引（0-based）。
    eps : float, default=0.001
        平滑因子，防止零频数导致除零。

    Attributes
    ----------
    woe_map_ : dict
        {col: {bin_idx: woe_value}} 各列各箱的 WOE 值。
    iv_values_ : dict
        {col: float} 各列的 IV 值。
    """

    def __init__(self, binner=None, eps=0.001):
        self.binner = binner
        self.eps = eps

    def fit(self, X, y=None):
        """
        计算各列各箱的 WOE 值和 IV 值。

        Args:
            X: pandas DataFrame（已分箱的整数索引 或 原始值 + binner）
            y: 目标变量（二分类 0/1）

        Returns:
            self
        """
        X = validate_dataframe(X)
        if y is None:
            raise ValueError("WoeEncoder 需要目标变量 y")
        y = pd.Series(y, index=X.index)

        # 如有 binner，先分箱
        if self.binner is not None:
            X = self.binner.transform(X)

        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]

        self.woe_map_ = {}
        self.iv_values_ = {}

        total_pos = float(y.sum())
        total_neg = float(len(y) - y.sum())

        for col in X.columns:
            col_woe = {}
            col_iv = 0.0
            bin_indices = sorted(X[col].dropna().unique())

            for bin_idx in bin_indices:
                mask = X[col] == bin_idx
                n_pos = float(y[mask].sum())
                n_neg = float(mask.sum() - n_pos)

                # 平滑后的分布
                dist_pos = (n_pos + self.eps) / (total_pos + self.eps * len(bin_indices))
                dist_neg = (n_neg + self.eps) / (total_neg + self.eps * len(bin_indices))

                woe = np.log(dist_pos / dist_neg)
                col_woe[bin_idx] = woe
                col_iv += (dist_pos - dist_neg) * woe

            self.woe_map_[col] = col_woe
            self.iv_values_[col] = col_iv

        return self

    def transform(self, X):
        """
        将箱索引替换为 WOE 值。

        Args:
            X: pandas DataFrame

        Returns:
            WOE 编码后的 DataFrame（浮点值）
        """
        validate_dataframe(X)

        # 如有 binner，先分箱
        if self.binner is not None:
            X = self.binner.transform(X)

        X_out = pd.DataFrame(index=X.index)

        for col in X.columns:
            if col not in self.woe_map_:
                raise ValueError(f"列 '{col}' 未在 fit 时训练")
            woe_map = self.woe_map_[col]
            X_out[col] = X[col].map(woe_map)

        return X_out

    def get_woe_table(self, feature):
        """
        获取指定特征的 WOE 明细表。

        Args:
            feature: 特征名

        Returns:
            DataFrame，包含 bin_index, woe, iv_contribution 等列
        """
        if feature not in self.woe_map_:
            raise ValueError(f"特征 '{feature}' 未训练")

        rows = []
        woe_map = self.woe_map_[feature]
        for bin_idx, woe_val in sorted(woe_map.items()):
            rows.append({
                "bin_index": bin_idx,
                "woe": woe_val,
            })
        return pd.DataFrame(rows)


class BinnerWoeEncoder(RiskTransformer):
    """
    分箱+WOE 联合算子：便捷的一步到位方案。

    内部组合 ChiMergeBinner + WoeEncoder，对外暴露两者的能力。

    Parameters
    ----------
    max_bins : int, default=10
        传递给内部 ChiMergeBinner。
    min_bins : int, default=2
    bin_pct_threshold : float, default=0.05
    confidence_level : float, default=0.9
    special_values : dict or None, default=None
    categorical_features : list[str] or None, default=None
    eps : float, default=0.001
        WOE 平滑因子。

    Attributes
    ----------
    binner_ : ChiMergeBinner
        拟合后的分箱器。
    encoder_ : WoeEncoder
        拟合后的 WOE 编码器。
    bin_edges_ : dict
        透传自 binner_。
    bin_labels_ : dict
        透传自 binner_。
    woe_map_ : dict
        透传自 encoder_。
    iv_values_ : dict
        透传自 encoder_。
    """

    def __init__(
        self,
        max_bins=10,
        min_bins=2,
        bin_pct_threshold=0.05,
        confidence_level=0.9,
        special_values=None,
        categorical_features=None,
        eps=0.001,
    ):
        self.max_bins = max_bins
        self.min_bins = min_bins
        self.bin_pct_threshold = bin_pct_threshold
        self.confidence_level = confidence_level
        self.special_values = special_values
        self.categorical_features = categorical_features
        self.eps = eps

    def fit(self, X, y=None):
        """
        先分箱，再 WOE 编码。

        Args:
            X: pandas DataFrame（原始值）
            y: 目标变量

        Returns:
            self
        """
        X = validate_dataframe(X)
        if y is None:
            raise ValueError("BinnerWoeEncoder 需要目标变量 y")

        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]

        # Step 1: 分箱
        self.binner_ = ChiMergeBinner(
            max_bins=self.max_bins,
            min_bins=self.min_bins,
            bin_pct_threshold=self.bin_pct_threshold,
            confidence_level=self.confidence_level,
            special_values=self.special_values,
            categorical_features=self.categorical_features,
        )
        X_binned = self.binner_.fit_transform(X, y)

        # Step 2: WOE 编码
        self.encoder_ = WoeEncoder(eps=self.eps)
        self.encoder_.fit(X_binned, y)

        # 透传属性
        self.bin_edges_ = self.binner_.bin_edges_
        self.bin_labels_ = self.binner_.bin_labels_
        self.woe_map_ = self.encoder_.woe_map_
        self.iv_values_ = self.encoder_.iv_values_

        return self

    def transform(self, X):
        """
        先分箱再 WOE 编码。

        Args:
            X: pandas DataFrame（原始值）

        Returns:
            WOE 编码后的 DataFrame
        """
        validate_dataframe(X)
        X_binned = self.binner_.transform(X)
        return self.encoder_.transform(X_binned)

    def get_bin_table(self, feature):
        """获取分箱汇总表（透传 binner）"""
        return self.binner_.get_bin_table(feature)

    def get_woe_table(self, feature):
        """获取 WOE 明细表（透传 encoder）"""
        return self.encoder_.get_woe_table(feature)
