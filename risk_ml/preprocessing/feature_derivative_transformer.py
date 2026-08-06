"""
特征衍生 pipeline 步骤 — FeatureDerivativeTransformer

把 feature_derivative 的四则运算表达式封装成 sklearn 兼容的 RiskTransformer，
可放进 RiskPipeline / sklearn Pipeline：
    pipe = RiskPipeline([
        ("fd", FeatureDerivativeTransformer({"ratio": "amount/income"})),
        ("cleaner", FeatureCleaner(...)),
        ...
    ])

关键设计：transform 使用 feature_derivative.transpile 转译出的 **numpy 源码**
执行（而非 df.eval），与线上部署内核跑同一份代码 → 构造级保证线上线下一致。
语义锚定 feature_derivative 的 Pandas 策略：
- 除零 → NaN、NaN 传播（根节点 isinf→NaN）
- fill_value：计算前对表达式涉及列做 NaN 预填充（只填 NaN 不填 inf）

限制（v1 明确决策）：
- 衍生列名不得与输入列或同步其他衍生列冲突（feature_derivative 允许覆盖，
  但部署链路要求列名唯一，fit 时报错拒绝）。
- 同一步内表达式互不可见（各自只看到 op 输入列）；链式衍生请拆成两个步骤。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.utils.validation import check_is_fitted

from feature_derivative import (
    MissingVariableError,
    extract_variables,
    to_numpy_source,
    validate_ast_safety,
)
from .._base import RiskTransformer, validate_dataframe

# 表达式输入形式：
#   {"target": "expr"}                        dict 形式
#   [("target", "expr"), ...]                 list 元组形式
#   [{"target": ..., "expression": ...}, ...] list dict 形式
ExpressionInput = Union[
    Dict[str, str],
    List[Tuple[str, str]],
    List[Dict[str, str]],
]


@dataclass
class DerivedSpec:
    """一个已编译的衍生表达式。

    不含可调用函数：exec 动态函数不可 pickle（pkl 参考打分链路会崩），
    执行函数统一走模块级 _get_fn(source) 缓存（与 online_deploy 的
    _get_derive_fn 同模式），spec 仅存源码字符串 → 天然可 pickle。
    """

    target: str          # 衍生列名
    expression: str      # 原始表达式（展示用）
    variables: List[str] # 表达式涉及的变量（按首次出现顺序）
    source: str          # 转译出的 numpy 源码


# 转译源码 → 可调用函数缓存（lazy exec，与 risk_ml/online_deploy/_ops.py 的
# _DERIVE_FNS 共用同一份源码，两侧语义一致）
_FD_FNS = {}


def _get_fn(source):
    fn = _FD_FNS.get(source)
    if fn is None:
        ns = {}
        exec(source, {"np": np}, ns)  # 白名单表达式 + 空命名空间，安全
        fn = ns["_fd"]
        _FD_FNS[source] = fn
    return fn


def _iter_expressions(expressions: ExpressionInput):
    """归一化不同输入形式 → (target, expression) 迭代器。"""
    if isinstance(expressions, dict):
        for target, expr in expressions.items():
            yield target, expr
    elif isinstance(expressions, (list, tuple)):
        for item in expressions:
            if isinstance(item, dict):
                yield item["target"], item["expression"]
            else:
                target, expr = item
                yield target, expr
    else:
        raise TypeError(
            f"expressions 必须是 dict / list[tuple] / list[dict]，"
            f"收到 {type(expressions).__name__}"
        )


class FeatureDerivativeTransformer(RiskTransformer):
    """
    特征衍生转换器：四则运算表达式 → 新特征列。

    Parameters
    ----------
    expressions : dict | list, 必须
        特征衍生表达式配置。三种形式：
        - ``{"ratio": "amount/income"}`` — dict，键为衍生列名
        - ``[("ratio", "amount/income"), ...]`` — 元组列表
        - ``[{"target": "ratio", "expression": "amount/income"}, ...]``
    fill_value : float or None, default=None
        缺失值预填充值。None = NaN 传播（默认）；数值 = 计算前对表达式
        涉及的列做 NaN → fill 填充（与 feature_derivative.transform 一致）。

    Attributes
    ----------
    expression_specs_ : list[DerivedSpec]
        编译后的衍生表达式（target / variables / numpy 源码 / 可调用函数）。

    Examples
    --------
    >>> fd = FeatureDerivativeTransformer({
    ...     "ratio": "amount/income",
    ...     "amount_1k": "amount / 1000",
    ... })
    >>> fd.fit(X)
    >>> fd.transform(X).columns  # 原列 + ratio + amount_1k
    """

    def __init__(self, expressions: Optional[ExpressionInput] = None,
                 fill_value: Optional[float] = None):
        self.expressions = expressions
        self.fill_value = fill_value

    # ------------------------------------------------------------------
    # fit：前置校验 + 转译（错误在训练期暴露，而非部署期）
    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        X = validate_dataframe(X)
        cols = X.columns.tolist()

        if not self.expressions:
            raise ValueError(
                "FeatureDerivativeTransformer 需要 expressions 参数"
                "（如 {'ratio': 'amount/income'}）"
            )

        specs: List[DerivedSpec] = []
        seen: set = set()
        for target, expression in _iter_expressions(self.expressions):
            # 1. AST 安全校验（ExpressionSyntaxError / UnsafeExpressionError）
            validate_ast_safety(expression)
            # 2. 变量提取 + 存在性校验
            variables = extract_variables(expression)
            missing = [v for v in variables if v not in cols]
            if missing:
                raise MissingVariableError(missing, cols)
            # 3. 撞名拒绝（部署链路要求列名唯一）
            if target in cols:
                raise ValueError(
                    f"衍生列名 {target!r} 与输入列冲突：feature_derivative 允许"
                    "覆盖原列，但部署链路要求列名唯一。请更换衍生列名"
                    f"（表达式 {expression!r}）"
                )
            if target in seen:
                raise ValueError(f"衍生列名 {target!r} 重复定义（表达式 {expression!r}）")
            seen.add(target)
            # 4. 转译（与部署内核同源码；执行函数由 _get_fn 按源码懒编译缓存）
            var_to_idx = {v: i for i, v in enumerate(variables)}
            source = to_numpy_source(expression, var_to_idx, fill_value=self.fill_value)
            specs.append(DerivedSpec(
                target=target, expression=expression,
                variables=variables, source=source,
            ))

        self.expression_specs_ = specs
        self.feature_names_in_ = cols
        self.n_features_in_ = X.shape[1]
        return self

    # ------------------------------------------------------------------
    # transform：同一份转译源码，numpy 向量化执行
    # ------------------------------------------------------------------
    def transform(self, X):
        check_is_fitted(self, "expression_specs_")
        X = validate_dataframe(X)
        if list(X.columns) != self.feature_names_in_:
            raise ValueError(
                f"输入列与 fit 时不一致（期望 {self.feature_names_in_}，"
                f"收到 {list(X.columns)}）。feature_derivative 不支持对"
                "列序变化的输入做 transform"
            )
        out = X.copy()
        for spec in self.expression_specs_:
            sub = np.asarray(X[spec.variables], dtype=np.float64)
            out[spec.target] = _get_fn(spec.source)(sub)
        return out

    # ------------------------------------------------------------------
    # sklearn set_output / Pipeline 需要：输出特征名 = 原列 + 衍生列
    # ------------------------------------------------------------------
    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "expression_specs_")
        return np.array(
            self.feature_names_in_ + [s.target for s in self.expression_specs_]
        )
