"""
自定义算子注册表 — to_deploy 协议

允许用户为自定义 transformer 算子注册部署构建器，扩展部署支持。

两种扩展方式：
1. to_deploy 协议：自定义算子实现 ``to_deploy(input_columns)`` 方法，
   返回一个 DeployOp 实例（或 None 表示不支持）。
2. register_deploy_builder：为某个算子类注册外部构建函数
   ``builder(step, input_columns) -> DeployOp``。
"""

from typing import Any, Callable, Dict, Optional, Type

from ._base import DeployOp

# 算子类型 → 构建函数注册表
_DEPLOY_BUILDERS: Dict[Type, Callable[[Any, list], Optional[DeployOp]]] = {}


def register_deploy_builder(step_cls: Type, builder: Callable[[Any, list], Optional[DeployOp]]):
    """为自定义算子类注册部署构建器。

    Args:
        step_cls: 自定义算子类（sklearn transformer）
        builder: builder(step, input_columns) -> DeployOp，返回 None 表示不支持

    Example
    -------
    >>> register_deploy_builder(MyWinsorizer, lambda step, cols: WinsorizeOp(step, cols))
    """
    _DEPLOY_BUILDERS[step_cls] = builder


def get_builder(step) -> Optional[Callable[[Any, list], Optional[DeployOp]]]:
    """按算子类型精确匹配构建器（不支持父类继承匹配，避免歧义）。"""
    return _DEPLOY_BUILDERS.get(type(step))


def build_deploy_op(step, input_columns) -> Optional[DeployOp]:
    """为给定 pipeline 步骤构建部署算子。

    优先级：注册表 > to_deploy 协议。
    返回 None 表示该步骤不支持部署（由 parser 决定如何报错）。

    Args:
        step: 已拟合的 pipeline 步骤
        input_columns: 该步骤的输入列名

    Returns:
        DeployOp 或 None
    """
    # 1. 注册表（用户显式注册优先）
    builder = get_builder(step)
    if builder is not None:
        return builder(step, input_columns)

    # 2. to_deploy 协议（算子自身实现）
    to_deploy = getattr(step, "to_deploy", None)
    if callable(to_deploy):
        try:
            result = to_deploy(input_columns)
        except TypeError:
            result = to_deploy()  # 兼容无参实现
        return result

    return None
