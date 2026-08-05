"""
编码算子抽象基类 — BaseEncoder

定义编码算子的通用接口与统一线上推理契约：

- 子类必须实现 ``fit(X, y)``，拟合后产出 ``woe_map_``（``{col: {code: woe}}``）
- 可选：若内嵌分箱，产出 post-fit 属性 ``binner_``（BaseBinner 实例）；
  也可沿用构造参数 ``binner``（如 ``WoeEncoder(binner=...)``）
- 基类提供统一 ``transform``：先（可选）分箱，再按 ``woe_map_`` 替换为编码值

这样新增编码算子只需继承 BaseEncoder 并实现 fit，部署端（WoeOp / BinWoeOp）
与线上推理（统一的 map 替换）均无需任何改动。
"""

from abc import ABC, abstractmethod

import pandas as pd

from .._base import RiskTransformer, validate_dataframe


class BaseEncoder(RiskTransformer, ABC):
    """
    编码算子抽象基类。

    Attributes
    ----------
    woe_map_ : dict
        {col: {code: woe_value}} 各列各箱/类别的编码映射，线上推理的单一来源。
    iv_values_ : dict (可选)
        {col: float} 各列 IV 值。
    """

    @abstractmethod
    def fit(self, X, y=None):
        """
        子类实现：拟合并产出 woe_map_。

        Args:
            X: pandas DataFrame
            y: 目标变量（二分类 0/1）

        Returns:
            self
        """
        raise NotImplementedError

    def transform(self, X):
        """
        统一编码 transform：先（可选）分箱，再按 woe_map_ 替换为编码值。

        分箱器来源（取其一）：
        - ``binner_``：post-fit 属性，编码器内嵌分箱器（如 BinnerWoeEncoder）
        - ``binner``  ：构造参数，外部传入的分箱器（如 WoeEncoder(binner=...)）

        Args:
            X: pandas DataFrame

        Returns:
            DataFrame，编码后的浮点值
        """
        validate_dataframe(X)

        binner = getattr(self, "binner_", None)
        if binner is None:
            binner = getattr(self, "binner", None)
        if binner is not None:
            X = binner.transform(X)

        X_out = pd.DataFrame(index=X.index)
        for col in X.columns:
            if col not in self.woe_map_:
                raise ValueError(f"列 '{col}' 未在 fit 时训练")
            X_out[col] = X[col].map(self.woe_map_[col])
        return X_out

    def get_woe_table(self, feature):
        """
        获取指定特征的 WOE 明细表。

        Args:
            feature: 特征名

        Returns:
            DataFrame，包含 bin_index, woe 列
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
