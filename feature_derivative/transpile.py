"""
表达式 → numpy 向量化源码转译器（部署用）

把经过 AST 白名单校验的四则运算表达式，转译成一段可直接 exec 的
numpy 向量化源码。这是 feature_derivative 集成进 online_deploy 的桥梁：
线下 transformer 与线上部署内核跑**同一份转译源码**，构造级保证线上线下一致。

设计要点：
- 输入必须是 AST 白名单子集（parser._SAFE_AST_NODES），白名单小 → 转译面极小。
- 变量绑定用 `_v{i}`（按 var_to_idx 中的索引），不依赖真实列名做标识符，
  天然免疫 Python 关键字 / 特殊字符列名。
- 除零：numpy 除法产生 ±inf，顶层统一 `np.isinf → NaN`，与 Pandas 策略
  （strategies.py:132 全部 inf 替换为 NaN）语义一致。
- fill_value：有值时对表达式涉及列做 `np.isnan → fill` 预填充（与三引擎一致，
  只填 NaN 不填 inf）。
- 根节点退化为纯常量（如 "2"）时用 np.full 保持 1-d 数组，避免 0-d 广播歧义。

用法（driver 侧 / transformer）:
    import numpy as np
    src = to_numpy_source("a/(a+b)", {"a": 0, "b": 1})
    ns = {}; exec(src, {"np": np}, ns)
    result = ns["_fd"](X[:, [0, 1]])   # X: (n, 2) float64
"""

import ast
from typing import Dict, List, Optional

import numpy as np

from .exceptions import FeatureDerivativeError
from .parser import extract_variables, validate_ast_safety


# 操作符 → numpy/Python 符号
_BIN_OP_SYMBOLS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
}
_UNARY_OP_SYMBOLS = {
    ast.USub: "-",
    ast.UAdd: "+",
}


def _emit(node, var_to_idx: Dict[str, int]) -> str:
    """递归把 AST 节点转译为 numpy 表达式字符串。

    仅接收已通过 validate_ast_safety 白名单校验的节点。
    """
    if isinstance(node, ast.Expression):
        return _emit(node.body, var_to_idx)
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Name):
        idx = var_to_idx.get(node.id)
        if idx is None:
            raise FeatureDerivativeError(
                f"表达式变量 {node.id!r} 不在 var_to_idx 映射中"
            )
        return f"_v{idx}"
    if isinstance(node, ast.BinOp):
        op = _BIN_OP_SYMBOLS.get(type(node.op))
        if op is None:
            raise FeatureDerivativeError(
                f"不支持的操作符: {type(node.op).__name__}（仅四则 + - * /）"
            )
        left = _emit(node.left, var_to_idx)
        right = _emit(node.right, var_to_idx)
        return f"({left} {op} {right})"
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OP_SYMBOLS.get(type(node.op))
        if op is None:
            raise FeatureDerivativeError(
                f"不支持的一元操作符: {type(node.op).__name__}（仅正负号）"
            )
        operand = _emit(node.operand, var_to_idx)
        return f"({op}{operand})"
    raise FeatureDerivativeError(
        f"无法转译的 AST 节点: {type(node).__name__}"
    )


def _is_constant_only(tree: ast.Expression) -> bool:
    """根节点是否为纯常量（如 "2"），需要 np.full 保持 1-d。"""
    return isinstance(tree.body, ast.Constant)


def to_numpy_source(
    expression: str,
    var_to_idx: Dict[str, int],
    fill_value: Optional[float] = None,
) -> str:
    """表达式 → numpy 向量化源码字符串。

    Args:
        expression: 四则运算表达式，如 "a/(a+b)"
        var_to_idx: {变量名: 变量子数组中的列位置}。_fd 收到的 X 是
                    (n, len(var_to_idx)) 的变量子数组，位置按调用方决定。
        fill_value: 缺失值预填充值。None = NaN 传播；数值 = 计算前对涉及列
                    做 np.isnan → fill（与 feature_derivative 三引擎一致）。

    Returns:
        定义 `_fd(X)` 的源码字符串。调用方 exec 后调用 ns["_fd"](X)，
        X 必须是 float64（numpy isinf/isnan 不支持 int 数组）。

    Raises:
        ExpressionSyntaxError / UnsafeExpressionError: 表达式未通过白名单校验
        FeatureDerivativeError: var_to_idx 缺失表达式变量
    """
    tree = validate_ast_safety(expression)
    variables = extract_variables(expression)

    missing = [v for v in variables if v not in var_to_idx]
    if missing:
        raise FeatureDerivativeError(
            f"var_to_idx 缺少表达式变量: {missing}（表达式 {expression!r}）"
        )

    lines = ["def _fd(X):"]
    for i, v in enumerate(variables):
        idx = var_to_idx[v]
        if fill_value is None:
            lines.append(f"    _v{i} = X[:, {idx}]")
        else:
            lines.append(
                f"    _v{i} = np.where(np.isnan(X[:, {idx}]), "
                f"{float(fill_value)!r}, X[:, {idx}])"
            )

    body = _emit(tree, var_to_idx)
    if _is_constant_only(tree):
        body = f"np.full(X.shape[0], {body})"
    lines.append(f"    _r = {body}")
    lines.append("    _r = np.where(np.isinf(_r), np.nan, _r)")
    lines.append("    return _r")
    return "\n".join(lines)


def compile_numpy_fn(
    expression: str,
    var_to_idx: Dict[str, int],
    fill_value: Optional[float] = None,
):
    """转译 + exec，直接返回可调用的 numpy 函数 _fd(X) -> np.ndarray(n,)。

    与 to_numpy_source 的源码一一对应，供 transformer / 测试直接使用。
    """
    source = to_numpy_source(expression, var_to_idx, fill_value)
    ns = {}
    exec(source, {"np": np}, ns)  # 白名单表达式 + 空命名空间，安全
    return ns["_fd"]
