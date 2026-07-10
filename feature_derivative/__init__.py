"""
特征衍生框架 (Feature Derivative Framework)
============================================

接收四则运算表达式字符串，自动解析变量，在三种计算引擎上高效生成新特征列。

支持的引擎：
- Pandas      : df.eval()  向量化计算，适合离线批处理
- PySpark     : F.expr()   分布式计算，适合大规模数据
- Python Dict : safe eval  单条计算，适合在线推理服务

快速开始:
    from feature_derivative import transform

    # Pandas
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = transform(df, "a/(a+b)", "ratio")

    # Dict（在线服务）
    result = transform({"a": 1, "b": 4}, "a/(a+b)", "ratio")
"""

from .context import FeatureDerivativeContext
from .exceptions import (
    FeatureDerivativeError,
    MissingVariableError,
    ExpressionSyntaxError,
    UnsafeExpressionError,
)
from .parser import extract_variables, validate_ast_safety
from .strategies import PandasStrategy, SparkStrategy, OnlineStrategy

# 全局上下文实例（单例，无需重复创建）
_context = FeatureDerivativeContext()


def transform(data, expression: str, target_col: str, fill_value=None):
    """
    统一的特征衍生入口函数。

    自动识别输入数据类型，选择对应计算引擎，执行表达式计算，
    将结果作为新列添加到数据中。

    Args:
        data: 输入数据 (pandas.DataFrame / pyspark.sql.DataFrame / dict)
        expression: 四则运算表达式字符串，如 "a/(a+b)"
        target_col: 新特征列名
        fill_value: 缺失值填充值，None=传播策略（默认），设为数值则预填充

    Returns:
        添加了新列的数据（类型与输入一致）

    Raises:
        MissingVariableError: 表达式变量在数据中缺失
        ExpressionSyntaxError: 表达式语法错误
        UnsafeExpressionError: 表达式不安全（仅 Online 模式）
        TypeError: 不支持的数据类型
    """
    return _context.transform(data, expression, target_col, fill_value=fill_value)


__all__ = [
    "transform",
    "FeatureDerivativeContext",
    "PandasStrategy",
    "SparkStrategy",
    "OnlineStrategy",
    "FeatureDerivativeError",
    "MissingVariableError",
    "ExpressionSyntaxError",
    "UnsafeExpressionError",
    "extract_variables",
    "validate_ast_safety",
]
