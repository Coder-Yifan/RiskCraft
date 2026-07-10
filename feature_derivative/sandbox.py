"""
安全沙箱 — Python 单变量模式的 eval() 防护

核心职责：
在受限环境中执行表达式求值，防止任意代码执行。

三层防护机制：
1. AST 节点白名单：在执行前校验表达式结构，拒绝函数调用、属性访问等
2. __builtins__ 置空：切断所有内置函数（open, exec, eval, __import__ 等）
3. locals 限定：只传入表达式所需的变量，不暴露任何额外命名空间

安全模型说明：
- 攻击向量 __import__('os').system('rm -rf /') 被 AST 白名单拦截
  （ast.Call / ast.Attribute 不在白名单中）
- 攻击向量 eval("__import__('os')") 同样被拦截
  （ast.Call 不在白名单中，且 __builtins__ 为空）
- 攻击向量 (1).__class__.__bases__[0].__subclasses__() 被拦截
  （ast.Attribute / ast.Subscript 不在白名单中）
"""

import ast

from .parser import validate_ast_safety
from .exceptions import UnsafeExpressionError


def safe_eval_expression(expression: str, variables: dict):
    """
    在安全沙箱中执行数学表达式求值。

    Args:
        expression: 数学表达式字符串，如 "a/(a+b)"
        variables: 变量名到值的映射，如 {"a": 1, "b": 4}

    Returns:
        计算结果。除以零时返回 None，None 参与运算时返回 None。

    Raises:
        UnsafeExpressionError: 表达式不安全（包含非法 AST 节点）

    示例:
        >>> safe_eval_expression("a/(a+b)", {"a": 1, "b": 4})
        0.2
        >>> safe_eval_expression("a/b", {"a": 1, "b": 0})
        None
    """
    # ── 第一层防护：AST 节点白名单校验 ──
    tree = validate_ast_safety(expression)

    # ── 第二层防护：__builtins__ 置空 ──
    # 彻底切断内置函数访问，即使 AST 校验被绕过也无法调用 open/exec 等
    restricted_globals = {"__builtins__": {}}

    # ── 第三层防护：locals 仅包含已验证的变量 ──
    restricted_locals = dict(variables)

    # ── 执行求值 ──
    try:
        result = eval(
            compile(tree, "<feature_derivative>", "eval"),
            restricted_globals,
            restricted_locals,
        )
        return result
    except ZeroDivisionError:
        # 除以零：返回 None 作为"无法计算"的信号
        return None
    except TypeError:
        # None 参与运算（如 None + 1）：传播为 None
        return None
