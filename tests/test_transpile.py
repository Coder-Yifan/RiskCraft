"""
转译器测试 — feature_derivative.transpile

核心验证：转译出的 numpy 源码在数值上等价于 feature_derivative 三引擎
（Pandas df.eval / Online 沙箱），语义锚定 Pandas 策略：
- 除零 → NaN（Pandas: inf→NaN；Online: 除零→None，二者收敛于缺失）
- NaN 传播（Online: None 参与运算 → None）
- fill_value 预填充（只填 NaN 不填 inf，与 strategies.py 一致）

运行方式:
    pytest tests/test_transpile.py -v
"""

import re

import numpy as np
import pandas as pd
import pytest

from feature_derivative import (
    ExpressionSyntaxError,
    FeatureDerivativeError,
    UnsafeExpressionError,
    compile_numpy_fn,
    to_numpy_source,
    transform,
)
from feature_derivative.sandbox import safe_eval_expression


def _arr(values):
    """list of dict → (n, 2) float64 数组（a, b 两列）。"""
    return np.array([[r["a"], r["b"]] for r in values], dtype=np.float64)


# ============================================================
# 源码结构
# ============================================================
class TestSourceStructure:
    def test_defines_fd(self):
        src = to_numpy_source("a/(a+b)", {"a": 0, "b": 1})
        assert "def _fd(X):" in src
        assert "_v0 = X[:, 0]" in src
        assert "_v1 = X[:, 1]" in src

    def test_uses_index_not_identifier(self):
        """变量绑定用 _v{i}，不依赖真实列名 → 关键字/特殊字符列名免疫。"""
        src = to_numpy_source("a+b", {"a": 0, "b": 1})
        assert not re.search(r"\ba\b", src)   # 没有裸变量名 a（np.nan 的 'a' 不受影响）
        assert "_v0 = X[:, 0]" in src
        assert "_v1 = X[:, 1]" in src

    def test_missing_var_mapping_raises(self):
        with pytest.raises(FeatureDerivativeError):
            to_numpy_source("a/(a+b)", {"a": 0})  # b 缺映射


# ============================================================
# 数值一致性 — 锚定 Pandas 引擎（transform）
# ============================================================
class TestPandasParity:
    """转译 numpy 结果 == feature_derivative.transform（df.eval）结果。"""

    # 注：纯常量表达式（如 "2"）不在 CASES —— Pandas 锚 df.eval("2") 返回
    # 标量 int32，strategies.py:132 的 .replace() 会崩（feature_derivative
    # 自身不支持常量根）；转译器支持，由 test_constant_root_is_1d 单独覆盖。
    CASES = [
        "a/(a+b)",
        "(a-b)*(a+b)",
        "a + b - a * b",
        "a/b + 1",
        "a + 2 * b",
        "-a + b",
        "a - -b",
        "a",
    ]

    @pytest.mark.parametrize("expr", CASES)
    def test_matches_pandas_engine(self, expr):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, np.nan, -1.0, 0.0],
            "b": [4.0, 5.0, np.nan, 6.0, 7.0, 0.0],
        })
        expected = transform(df, expr, "t")["t"]
        fn = compile_numpy_fn(expr, {"a": 0, "b": 1})
        result = fn(df[["a", "b"]].to_numpy(dtype=np.float64))
        assert result.shape == (len(df),)
        np.testing.assert_allclose(result, expected.to_numpy(),
                                   rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_div_zero_is_nan(self):
        """除零 → NaN（与 Pandas 策略 inf→NaN 一致）。"""
        df = pd.DataFrame({"a": [1.0, -1.0], "b": [0.0, 0.0]})
        expected = transform(df, "a/b", "t")["t"]
        fn = compile_numpy_fn("a/b", {"a": 0, "b": 1})
        result = fn(df[["a", "b"]].to_numpy(dtype=np.float64))
        assert np.isnan(result).all()
        np.testing.assert_array_equal(result, expected.to_numpy())

    def test_nan_propagation(self):
        """NaN 参与运算 → NaN（与 Pandas NaN 传播一致）。"""
        df = pd.DataFrame({"a": [np.nan, 2.0], "b": [3.0, 4.0]})
        fn = compile_numpy_fn("a*b", {"a": 0, "b": 1})
        result = fn(df[["a", "b"]].to_numpy(dtype=np.float64))
        assert np.isnan(result[0]) and not np.isnan(result[1])

    def test_fill_value_matches(self):
        """fill_value 预填充与 transform(fill_value=...) 一致。"""
        df = pd.DataFrame({"a": [np.nan, 2.0], "b": [0.0, np.nan]})
        expected = transform(df, "a/b", "t", fill_value=1.0)["t"]
        fn = compile_numpy_fn("a/b", {"a": 0, "b": 1}, fill_value=1.0)
        result = fn(df[["a", "b"]].to_numpy(dtype=np.float64))
        np.testing.assert_allclose(result, expected.to_numpy(), equal_nan=True)

    def test_fill_only_nan_not_inf(self):
        """fill_value 只填 NaN 不填 inf（与 strategies.py:122 一致）。

        输入 inf 不被 fill 替换，但顶层 isinf→NaN（strategies.py:132 替换
        所有 inf）会把它转成 NaN —— 与 Pandas 锚完全一致。
        """
        df = pd.DataFrame({"a": [np.inf, 1.0], "b": [1.0, np.inf]})
        expected = transform(df, "a + b", "t", fill_value=0.0)["t"]
        fn = compile_numpy_fn("a + b", {"a": 0, "b": 1}, fill_value=0.0)
        result = fn(df[["a", "b"]].to_numpy(dtype=np.float64))
        np.testing.assert_array_equal(result, expected.to_numpy())  # 全 NaN

    def test_constant_root_is_1d(self):
        """根节点为纯常量 → 1-d 数组（长度 = 行数）。"""
        fn = compile_numpy_fn("2", {"a": 0})
        X = np.zeros((5, 1))
        result = fn(X)
        assert result.shape == (5,)
        assert (result == 2.0).all()


# ============================================================
# 数值一致性 — Online 引擎（safe_eval_expression，单条）
# ============================================================
class TestOnlineParity:
    def test_matches_online_engine(self):
        """单条 dict 上，numpy 结果 == 沙箱 eval（None/NaN 均视为缺失）。"""
        expr = "a/(a+b)"
        rows = [
            {"a": 1.0, "b": 4.0},       # 正常
            {"a": None, "b": 4.0},      # None 传播 → 缺失
            {"a": 1.0, "b": 0.0},       # 除零 → 缺失
            {"a": 0.0, "b": 0.0},       # 0/0 → 缺失
            {"a": 2.0, "b": 6.0},       # 正常
        ]
        fn = compile_numpy_fn(expr, {"a": 0, "b": 1})
        X = _arr([{k: (v if v is not None else np.nan) for k, v in r.items()}
                  for r in rows])
        result = fn(X)
        for i, row in enumerate(rows):
            expected = safe_eval_expression(expr, row)  # float 或 None
            if expected is None:
                assert np.isnan(result[i]), f"row {i} 应缺失"
            else:
                assert np.isclose(result[i], expected), f"row {i} 应 {expected}"


# ============================================================
# 安全：转译只接受白名单表达式
# ============================================================
class TestSafety:
    def test_rejects_unsafe(self):
        with pytest.raises(UnsafeExpressionError):
            compile_numpy_fn("__import__('os').system('x')", {})
        with pytest.raises(UnsafeExpressionError):
            compile_numpy_fn("a ** 2", {"a": 0})          # 幂不在白名单
        with pytest.raises(UnsafeExpressionError):
            compile_numpy_fn("a > b", {"a": 0, "b": 1})   # 比较不在白名单

    def test_rejects_syntax_error(self):
        with pytest.raises(ExpressionSyntaxError):
            compile_numpy_fn("a +", {"a": 0})
