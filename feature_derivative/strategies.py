"""
策略模式实现 — 三种计算引擎的策略类

设计模式：策略模式（Strategy Pattern）
- BaseStrategy    : 抽象基类，定义统一接口
- PandasStrategy  : Pandas 引擎，利用 df.eval() 向量化计算
- SparkStrategy   : PySpark 引擎，利用 F.expr() 分布式计算
- OnlineStrategy  : Python 单变量引擎，利用安全沙箱 eval() 计算

缺失值处理策略（详见 README.md）：
- Pandas : NaN 传播（输入 NaN → 输出 NaN），除以零 inf → NaN
- PySpark: null 传播（ANSI SQL 标准行为），除以零 → null
- Online : None 传播（None 参与运算 → 返回 None），除以零 → None
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import pandas as pd

from .parser import extract_variables, validate_ast_safety
from .sandbox import safe_eval_expression
from .exceptions import MissingVariableError, UnsafeExpressionError


class BaseStrategy(ABC):
    """策略基类 — 定义统一的计算接口和校验逻辑"""

    @abstractmethod
    def compute(
        self,
        data: Any,
        expression: str,
        target_col: str,
        fill_value: Optional[float] = None,
    ) -> Any:
        """
        执行特征衍生计算。

        Args:
            data: 输入数据（具体类型由子类决定）
            expression: 数学表达式字符串，如 "a/(a+b)"
            target_col: 新特征列名，如 "new_feature"
            fill_value: 缺失值填充值。
                        None（默认）= NaN/None 传播；
                        设为具体值（如 0）= 缺失值预填充。

        Returns:
            添加了新列的数据（类型与输入一致）
        """
        pass

    @staticmethod
    def validate_expression(expression: str) -> None:
        """
        校验表达式的安全性（AST 节点白名单）。

        必须在 validate_variables() 之前调用，
        否则 extract_variables() 可能从不安全表达式中提取出
        __import__ 等非法变量名，导致抛出 MissingVariableError
        而非更准确的 UnsafeExpressionError。

        Args:
            expression: 数学表达式

        Raises:
            UnsafeExpressionError: 表达式包含不安全操作
            ExpressionSyntaxError: 表达式语法错误
        """
        validate_ast_safety(expression)

    @staticmethod
    def validate_variables(expression: str, available_vars: list) -> None:
        """
        校验表达式中的变量是否全部存在于数据中。

        前置条件：表达式已通过 validate_expression() 安全校验。

        Args:
            expression: 数学表达式
            available_vars: 数据中可用的列名/键名列表

        Raises:
            MissingVariableError: 存在缺失变量，异常消息中包含缺失字段名
        """
        required = extract_variables(expression)
        missing = [v for v in required if v not in available_vars]
        if missing:
            raise MissingVariableError(missing, available_vars)


class PandasStrategy(BaseStrategy):
    """
    Pandas 计算策略

    利用 df.eval() 实现向量化高性能计算。
    - 缺失值：NaN 传播（输入中的 NaN 在计算中自然传播）
    - 除以零：df.eval() 产生 inf，统一替换为 NaN
    - fill_value：若指定，计算前将参与运算列的 NaN 填充为该值
    """

    def compute(
        self,
        data: pd.DataFrame,
        expression: str,
        target_col: str,
        fill_value: Optional[float] = None,
    ) -> pd.DataFrame:
        # 1. 安全校验（必须在变量校验之前，防止不安全表达式被误报为变量缺失）
        self.validate_expression(expression)

        # 2. 变量校验
        self.validate_variables(expression, data.columns.tolist())

        # 3. 可选：缺失值预填充
        if fill_value is not None:
            required = extract_variables(expression)
            data = data.copy()
            for col in required:
                if col in data.columns:
                    data[col] = data[col].fillna(fill_value)
        else:
            data = data.copy()

        # 3. 使用 df.eval() 进行向量化计算
        # Pandas eval() 天然支持 NaN 传播：任何 NaN 参与运算结果为 NaN
        with np.errstate(divide="ignore", invalid="ignore"):
            result = data.eval(expression)

        # 4. 处理除以零产生的 inf / -inf，统一替换为 NaN
        result = result.replace([np.inf, -np.inf], np.nan)

        # 5. 将结果作为新列写入
        data[target_col] = result

        return data


class SparkStrategy(BaseStrategy):
    """
    PySpark 计算策略

    利用 F.expr() 实现分布式计算。
    - 缺失值：null 传播（Spark SQL ANSI 标准，null 参与运算 → null）
    - 除以零：Spark 非 ANSI 模式下返回 null（与 null 传播一致）
    - fill_value：若指定，计算前将参与运算列的 null 填充为该值

    注意：需要 Spark 3.0+ 环境，且 spark.sql.ansi.enabled=false（默认）
    """

    def compute(
        self,
        data: Any,
        expression: str,
        target_col: str,
        fill_value: Optional[float] = None,
    ) -> Any:
        from pyspark.sql import functions as F
        from pyspark.sql.types import FloatType

        # 1. 安全校验
        self.validate_expression(expression)

        # 2. 变量校验
        self.validate_variables(expression, data.columns)

        # 3. 可选：缺失值预填充
        if fill_value is not None:
            required = extract_variables(expression)
            # 构建 fillna 映射：只填充表达式涉及的列
            fill_map = {col: fill_value for col in required if col in data.columns}
            if fill_map:
                data = data.fillna(fill_map)

        # 3. 使用 F.expr() 进行分布式 SQL 表达式计算
        # Spark SQL 天然处理 null 传播和除以零（返回 null）
        data = data.withColumn(target_col, F.expr(expression).cast(FloatType()))

        return data


class OnlineStrategy(BaseStrategy):
    """
    Python 单变量 / 在线服务计算策略

    利用安全沙箱 eval() 实现单条请求计算。
    - 缺失值：None 参与运算会触发 TypeError，被捕获后返回 None
    - 除以零：ZeroDivisionError 被捕获，返回 None
    - fill_value：若指定，None 值在计算前被替换为该值

    安全性保障：
    - AST 节点白名单校验（仅允许四则运算节点）
    - __builtins__ 置空
    - locals 仅包含表达式所需变量
    """

    def compute(
        self,
        data: dict,
        expression: str,
        target_col: str,
        fill_value: Optional[float] = None,
    ) -> dict:
        # 1. 安全校验（必须在变量校验之前，拦截注入攻击）
        self.validate_expression(expression)

        # 2. 变量校验
        self.validate_variables(expression, list(data.keys()))

        # 3. 提取表达式所需的变量值
        required = extract_variables(expression)
        variables = {}
        for var in required:
            val = data[var]  # 校验已通过，key 必然存在
            if fill_value is not None and val is None:
                # 预填充模式：将 None 替换为指定值
                variables[var] = fill_value
            else:
                # 传播模式：None 值保留，后续 eval 会触发 TypeError → 返回 None
                variables[var] = val

        # 3. 在安全沙箱中执行计算
        result = safe_eval_expression(expression, variables)

        # 4. 返回添加了新字段的字典（不修改原始字典）
        output = dict(data)
        output[target_col] = result

        return output
