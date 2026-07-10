"""
上下文类 — 统一对外暴露的接口

核心职责：
1. 根据输入数据类型自动识别应使用的计算引擎
2. 将计算委托给对应的策略类
3. 对外提供统一的 transform() 调用入口

自动识别规则：
- pandas.DataFrame → PandasStrategy
- pyspark.sql.DataFrame → SparkStrategy
- dict → OnlineStrategy
- 其他类型 → 抛出 TypeError

PySpark 的延迟加载：
SparkStrategy 在实际使用时才导入 PySpark，
避免在无 Spark 环境的机器上因 import 失败而阻断整个框架。
"""

from typing import Optional

import pandas as pd

from .strategies import PandasStrategy, SparkStrategy, OnlineStrategy, BaseStrategy
from .exceptions import FeatureDerivativeError


class FeatureDerivativeContext:
    """
    特征衍生上下文（Context）

    策略模式中的"上下文"角色：
    - 持有策略引用，将请求委托给当前策略
    - 根据数据类型自动选择策略，调用方无需关心底层引擎
    """

    # 延迟加载的 PySpark DataFrame 类引用
    _spark_df_class = None

    @classmethod
    def _get_spark_df_class(cls):
        """
        延迟加载 PySpark DataFrame 类。

        避免在模块导入时就要求 PySpark 可用，
        只在用户实际传入 Spark DataFrame 时才尝试导入。
        """
        if cls._spark_df_class is None:
            try:
                from pyspark.sql import DataFrame as SparkDataFrame

                cls._spark_df_class = SparkDataFrame
            except ImportError:
                # PySpark 未安装，设为 None 占位
                cls._spark_df_class = None
        return cls._spark_df_class

    @classmethod
    def _detect_strategy(cls, data) -> BaseStrategy:
        """
        自动检测数据类型并返回对应的策略实例。

        Args:
            data: 输入数据

        Returns:
            对应的计算策略实例

        Raises:
            TypeError: 不支持的数据类型
        """
        # Pandas DataFrame
        if isinstance(data, pd.DataFrame):
            return PandasStrategy()

        # PySpark DataFrame（延迟检测）
        SparkDataFrame = cls._get_spark_df_class()
        if SparkDataFrame is not None and isinstance(data, SparkDataFrame):
            return SparkStrategy()

        # Python dict（单条请求）
        if isinstance(data, dict):
            return OnlineStrategy()

        # 不支持的类型
        raise TypeError(
            f"不支持的数据类型: {type(data).__name__}。"
            f"仅支持: pandas.DataFrame, pyspark.sql.DataFrame, dict。"
        )

    def transform(
        self,
        data,
        expression: str,
        target_col: str,
        fill_value: Optional[float] = None,
    ):
        """
        统一的特征衍生入口。

        自动识别数据类型 → 选择计算引擎 → 校验变量 → 执行计算 → 返回结果。

        Args:
            data: 输入数据（pandas.DataFrame / pyspark.sql.DataFrame / dict）
            expression: 四则运算表达式字符串，如 "a/(a+b)"
            target_col: 新特征列名，如 "new_feature"
            fill_value: 缺失值填充值。
                        None（默认）= NaN/None 传播策略；
                        设为具体值（如 0）= 缺失值预填充策略。

        Returns:
            添加了新列的数据（类型与输入一致）

        Raises:
            MissingVariableError: 表达式变量在数据中缺失
            ExpressionSyntaxError: 表达式语法错误
            UnsafeExpressionError: 表达式不安全（仅 Online 模式）
            TypeError: 不支持的数据类型
        """
        strategy = self._detect_strategy(data)
        return strategy.compute(data, expression, target_col, fill_value=fill_value)
