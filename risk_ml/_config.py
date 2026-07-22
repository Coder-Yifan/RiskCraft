"""
risk_ml 项目级配置

集中管理风控建模中的通用常量，确保全项目行为一致。
其他模块通过 from risk_ml._config import XXX 访问。
"""

import numpy as np

# ============================================================
# 缺失值配置
# ============================================================

# 缺失值哨兵值：原始数据中这些值代表"缺失"
# 风控场景常见约定：-999 表示一般缺失，-9998 表示拒绝披露，-9996 表示系统默认值
MISSING_VALUE_SENTINELS = [-999, -9998, -9996]

# 缺失值统一映射目标：所有哨兵值在进入模型前统一转为 np.nan
MISSING_VALUE_NAN = np.nan


def map_sentinels_to_nan(X, sentinels=None):
    """
    将 DataFrame 中的哨兵缺失值统一映射为 np.nan。

    该函数应在数据进入任何算子之前调用，
    确保后续所有组件只需处理 np.nan 一种缺失表示。

    Args:
        X: pandas DataFrame
        sentinels: 哨兵值列表，默认使用 MISSING_VALUE_SENTINELS

    Returns:
        映射后的 DataFrame（副本，不修改原始数据）

    Example:
        >>> from risk_ml._config import map_sentinels_to_nan
        >>> df = pd.DataFrame({"a": [-999, 1, 2], "b": [-9998, 5, -9996]})
        >>> map_sentinels_to_nan(df)
             a    b
        0  NaN  NaN
        1  1.0  5.0
        2  2.0  NaN
    """
    import pandas as pd

    if sentinels is None:
        sentinels = MISSING_VALUE_SENTINELS

    X = X.copy()
    # 仅对数值列执行替换（object 列不含数值哨兵）
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        X[col] = X[col].replace(sentinels, np.nan)
    return X
