"""
特征衍生框架 — 自定义异常类

提供细粒度的异常体系，帮助调用方精确定位问题：
- MissingVariableError : 表达式变量在数据中缺失
- ExpressionSyntaxError: 表达式语法不合法
- UnsafeExpressionError: 表达式包含不安全操作（仅 Online 模式触发）
"""

from typing import List, Optional


class FeatureDerivativeError(Exception):
    """特征衍生框架基础异常，所有自定义异常的父类"""
    pass


class MissingVariableError(FeatureDerivativeError):
    """
    表达式中的变量在输入数据中缺失。

    示例:
        表达式 "a/(a+b+c)" 中需要变量 c，
        但数据只有列 ["a", "b"]，则抛出此异常。
    """

    def __init__(self, missing_vars: List[str], available: Optional[List[str]] = None):
        self.missing_vars = missing_vars
        self.available = available or []
        msg = (
            f"表达式中的变量 {missing_vars} 在输入数据中缺失。"
            f" 当前可用字段: {self.available}"
        )
        super().__init__(msg)


class ExpressionSyntaxError(FeatureDerivativeError):
    """表达式语法不合法，无法被 Python ast 模块解析"""
    pass


class UnsafeExpressionError(FeatureDerivativeError):
    """
    表达式包含不安全操作。

    在 Online 模式下，仅允许纯四则运算 AST 节点；
    任何函数调用、属性访问、导入操作等都会触发此异常。
    """
    pass
