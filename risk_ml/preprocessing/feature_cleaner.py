"""
特征清洗算子 — FeatureCleaner

处理缺失值、异常值、常数列、低方差列等数据质量问题。
继承 RiskTransformer，完全 sklearn 兼容。

缺失值哨兵映射：在清洗前自动将配置文件中的哨兵值（如 -999）
统一映射为 np.nan，确保后续逻辑只需处理一种缺失表示。
"""

import warnings

import numpy as np
import pandas as pd

from .._base import RiskTransformer, validate_dataframe
from .._config import MISSING_VALUE_SENTINELS, map_sentinels_to_nan


class FeatureCleaner(RiskTransformer):
    """
    特征清洗算子：缺失值哨兵映射、缺失值填充、异常值截断、低质量列删除。

    处理流程：
    1. 哨兵值映射：将 sentinels（默认 [-999, -9998, -9996]）统一替换为 np.nan
    2. 删除高缺失率列、常数列、低方差列
    3. 填充剩余缺失值
    4. 处理异常值

    Parameters
    ----------
    sentinels : list or None, default=None
        缺失值哨兵值列表。原始数据中这些值代表"缺失"，
        在清洗前统一映射为 np.nan。
        None 表示使用项目配置 MISSING_VALUE_SENTINELS（[-999, -9998, -9996]）。
        设为空列表 [] 可禁用哨兵映射。
    missing_threshold : float, default=0.95
        缺失率 >= 此阈值的列将被删除（哨兵映射后计算缺失率）。
    missing_strategy : str, default='median'
        剩余缺失值的填充策略。
        'median' — 中位数填充
        'mean'   — 均值填充
        'constant' — 用 missing_fill_value 填充
        'drop_row'  — 删除含缺失值的行（仅 transform 时生效）
    missing_fill_value : float or None, default=None
        当 missing_strategy='constant' 时的填充值。
    outlier_method : str or None, default=None
        异常值检测方法。
        None — 不处理异常值
        'iqr' — 四分位距法
        'percentile' — 百分位截断法
    outlier_bounds : tuple, default=(0.01, 0.99)
        当 outlier_method='percentile' 时的上下界分位数。
    outlier_iqr_factor : float, default=1.5
        当 outlier_method='iqr' 时的 IQR 倍数。
    outlier_action : str, default='clip'
        异常值处理方式。
        'clip'    — 截断到边界值
        'set_nan' — 设为 NaN
    variance_threshold : float, default=0.0
        方差 <= 此阈值的列将被删除。
    nunique_threshold : int, default=1
        唯一值数 <= 此阈值的列（常数列）将被删除。

    Attributes
    ----------
    feature_names_in_ : list[str]
        fit 时输入的特征名。
    n_features_in_ : int
        fit 时输入的特征数。
    drop_columns_ : list[str]
        被删除的列名列表。
    impute_values_ : dict
        各列的缺失填充值 {col: value}。
    clip_bounds_ : dict
        各列的异常值截断边界 {col: (lower, upper)}。
    """

    def __init__(
        self,
        sentinels=None,
        missing_threshold=0.95,
        missing_strategy="median",
        missing_fill_value=None,
        outlier_method=None,
        outlier_bounds=(0.01, 0.99),
        outlier_iqr_factor=1.5,
        outlier_action="clip",
        variance_threshold=0.0,
        nunique_threshold=1,
    ):
        self.sentinels = sentinels
        self.missing_threshold = missing_threshold
        self.missing_strategy = missing_strategy
        self.missing_fill_value = missing_fill_value
        self.outlier_method = outlier_method
        self.outlier_bounds = outlier_bounds
        self.outlier_iqr_factor = outlier_iqr_factor
        self.outlier_action = outlier_action
        self.variance_threshold = variance_threshold
        self.nunique_threshold = nunique_threshold

    def _apply_sentinel_mapping(self, X):
        """
        将哨兵缺失值统一映射为 np.nan。

        使用项目配置的 MISSING_VALUE_SENTINELS 或用户自定义列表。
        仅对数值列执行替换。

        Args:
            X: pandas DataFrame

        Returns:
            映射后的 DataFrame（副本）
        """
        sentinels = self.sentinels if self.sentinels is not None else MISSING_VALUE_SENTINELS
        if not sentinels:
            return X.copy()
        return map_sentinels_to_nan(X, sentinels=sentinels)

    def fit(self, X, y=None):
        """
        学习缺失填充值、异常值边界和待删除列。

        处理流程：
        1. 哨兵值映射 → np.nan
        2. 识别高缺失率列、常数列、低方差列
        3. 计算缺失填充值
        4. 计算异常值截断边界

        Args:
            X: pandas DataFrame
            y: 忽略，保持 sklearn 接口兼容

        Returns:
            self
        """
        X = validate_dataframe(X)
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]

        # --- 0. 哨兵值映射 ---
        X = self._apply_sentinel_mapping(X)

        # --- 1. 识别需删除的列 ---
        drop_cols = set()

        # 高缺失率列
        missing_rates = X.isnull().mean()
        high_missing = missing_rates[missing_rates >= self.missing_threshold].index.tolist()
        drop_cols.update(high_missing)

        # 常数列（唯一值 <= nunique_threshold，排除 NaN）
        for col in X.columns:
            if col in drop_cols:
                continue
            nunique = X[col].dropna().nunique()
            if nunique <= self.nunique_threshold:
                drop_cols.add(col)

        # 低方差列（仅数值列，跳过 object/category 列）
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in drop_cols:
                continue
            try:
                if X[col].var() <= self.variance_threshold:
                    drop_cols.add(col)
            except TypeError:
                # 某些数值列可能含混合类型，跳过
                pass
                continue
            if X[col].var() <= self.variance_threshold:
                drop_cols.add(col)

        self.drop_columns_ = sorted(drop_cols)

        # --- 2. 计算保留列的缺失填充值 ---
        keep_cols = [c for c in X.columns if c not in drop_cols]
        numeric_keep = X[keep_cols].select_dtypes(include=[np.number]).columns.tolist()
        self.impute_values_ = {}
        for col in keep_cols:
            col_data = X[col].dropna()
            if col_data.empty:
                self.impute_values_[col] = 0
                continue

            # 非数值列：median/mean 无意义，使用众数
            if col not in numeric_keep:
                if self.missing_strategy in ("median", "mean"):
                    self.impute_values_[col] = col_data.mode().iloc[0]
                elif self.missing_strategy == "constant":
                    self.impute_values_[col] = self.missing_fill_value or 0
                elif self.missing_strategy == "drop_row":
                    self.impute_values_[col] = None
                continue

            if self.missing_strategy == "median":
                self.impute_values_[col] = col_data.median()
            elif self.missing_strategy == "mean":
                self.impute_values_[col] = col_data.mean()
            elif self.missing_strategy == "constant":
                self.impute_values_[col] = self.missing_fill_value or 0
            elif self.missing_strategy == "drop_row":
                self.impute_values_[col] = None  # 不填充，标记为行删除
            else:
                raise ValueError(
                    f"不支持的 missing_strategy: '{self.missing_strategy}'，"
                    f"可选: 'median', 'mean', 'constant', 'drop_row'"
                )

        # --- 3. 计算异常值截断边界（仅数值列） ---
        self.clip_bounds_ = {}
        if self.outlier_method is not None:
            numeric_keep = [c for c in keep_cols if c in numeric_cols]
            for col in numeric_keep:
                col_data = X[col].dropna()
                if col_data.empty:
                    continue

                if self.outlier_method == "percentile":
                    lower = col_data.quantile(self.outlier_bounds[0])
                    upper = col_data.quantile(self.outlier_bounds[1])
                elif self.outlier_method == "iqr":
                    q1 = col_data.quantile(0.25)
                    q3 = col_data.quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - self.outlier_iqr_factor * iqr
                    upper = q3 + self.outlier_iqr_factor * iqr
                else:
                    raise ValueError(
                        f"不支持的 outlier_method: '{self.outlier_method}'，"
                        f"可选: None, 'percentile', 'iqr'"
                    )

                self.clip_bounds_[col] = (lower, upper)

        return self

    def transform(self, X):
        """
        应用清洗：哨兵映射 → 删除列 → 填充缺失 → 处理异常值。

        Args:
            X: pandas DataFrame

        Returns:
            清洗后的 DataFrame
        """
        validate_dataframe(X)

        # 0. 哨兵值映射
        X = self._apply_sentinel_mapping(X)

        # 1. 删除标记列
        drop_present = [c for c in self.drop_columns_ if c in X.columns]
        if drop_present:
            X = X.drop(columns=drop_present)

        # 2. 填充缺失值
        for col, fill_val in self.impute_values_.items():
            if col not in X.columns:
                continue
            if fill_val is not None:
                X[col] = X[col].fillna(fill_val)
            # fill_val=None 表示 drop_row 策略，不在此填充

        # drop_row 策略：删除含缺失值的行
        if self.missing_strategy == "drop_row":
            keep_cols = [c for c in X.columns if c in self.impute_values_]
            X = X.dropna(subset=keep_cols)

        # 3. 处理异常值
        for col, (lower, upper) in self.clip_bounds_.items():
            if col not in X.columns:
                continue
            if self.outlier_action == "clip":
                X[col] = X[col].clip(lower=lower, upper=upper)
            elif self.outlier_action == "set_nan":
                mask = (X[col] < lower) | (X[col] > upper)
                X.loc[mask, col] = np.nan
            else:
                raise ValueError(
                    f"不支持的 outlier_action: '{self.outlier_action}'，"
                    f"可选: 'clip', 'set_nan'"
                )

        return X
