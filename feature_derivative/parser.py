"""
表达式解析器 — 变量提取 & AST 安全校验

核心职责：
1. extract_variables() : 从四则运算表达式中提取所有变量名
2. validate_ast_safety(): 校验 AST 节点白名单，防止注入攻击

设计原则：
- 使用 Python 标准库 ast 模块进行语法级解析，比正则更准确
- AST 节点白名单机制：只允许四则运算相关节点，其余一律拒绝
"""

import ast
from typing import List

from .exceptions import ExpressionSyntaxError, UnsafeExpressionError


# ============================================================
# AST 节点白名单 — 仅允许四则运算（+ - * /）和括号
# ============================================================
_SAFE_AST_NODES = {
    # --- 结构节点 ---
    ast.Expression,  # eval 模式的顶层节点
    ast.Load,        # 变量加载上下文
    # --- 运算节点 ---
    ast.BinOp,       # 二元运算 (a + b)
    ast.UnaryOp,     # 一元运算 (-a)
    # --- 操作符 ---
    ast.Add,         # +
    ast.Sub,         # -
    ast.Mult,        # *
    ast.Div,         # /
    ast.USub,        # 负号 -
    ast.UAdd,        # 正号 +
    # --- 叶子节点 ---
    ast.Name,        # 变量名 (a, b, c)
    ast.Constant,    # 常量 (Python 3.8+)
}

# Python 3.7 兼容：ast.Num 在 3.8 中被 ast.Constant 取代（3.14 移除）
try:
    _SAFE_AST_NODES.add(ast.Num)  # type: ignore[attr-defined]
except AttributeError:
    pass


def extract_variables(expression: str) -> List[str]:
    """
    从表达式中提取所有变量名（保持首次出现顺序，去重）。

    利用 ast 模块解析表达式，遍历所有 Name 节点收集变量名。
    比正则表达式更准确 — 不会误匹配字符串字面量或注释。

    Args:
        expression: 数学表达式字符串，如 "a/(a+b)"

    Returns:
        去重后的变量名列表，如 ["a", "b"]

    Raises:
        ExpressionSyntaxError: 表达式语法无效

    示例:
        >>> extract_variables("a/(a+b)")
        ['a', 'b']
        >>> extract_variables("((x+y)*z)/(x-y)")
        ['x', 'y', 'z']
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ExpressionSyntaxError(
            f"表达式语法错误: '{expression}' — {e}"
        )

    # 收集所有 Name 节点及其源码位置（lineno, col_offset）
    name_nodes: List[ast.Name] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            name_nodes.append(node)

    # 按源码位置排序，保证变量名按首次出现顺序排列
    name_nodes.sort(key=lambda n: (n.lineno, n.col_offset))

    # 去重，保持首次出现顺序
    variables: List[str] = []
    seen: set = set()
    for node in name_nodes:
        if node.id not in seen:
            seen.add(node.id)
            variables.append(node.id)

    return variables


def validate_ast_safety(expression: str) -> ast.AST:
    """
    校验表达式的 AST 是否安全，仅允许四则运算节点。

    这是 Online 模式 eval() 安全防护的第一道防线：
    在执行 eval() 之前，先检查 AST 中是否存在不在白名单中的节点。
    例如 __import__('os') 会产生 ast.Call 节点，不在白名单内，直接拒绝。

    Args:
        expression: 数学表达式字符串

    Returns:
        解析后的 AST 树（可用于后续编译执行）

    Raises:
        ExpressionSyntaxError: 表达式语法无效
        UnsafeExpressionError: 表达式包含不安全操作

    示例:
        >>> validate_ast_safety("a + b")  # 安全，正常返回 AST
        >>> validate_ast_safety("__import__('os')")  # 抛出 UnsafeExpressionError
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ExpressionSyntaxError(
            f"表达式语法错误: '{expression}' — {e}"
        )

    # 遍历 AST 所有节点，检查是否在白名单中
    for node in ast.walk(tree):
        node_type = type(node)
        if node_type not in _SAFE_AST_NODES:
            raise UnsafeExpressionError(
                f"表达式包含不允许的操作: {node_type.__name__} "
                f"(在 '{expression}' 中)。仅支持四则运算 (+, -, *, /)。"
            )

    return tree
