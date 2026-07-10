"""
特征衍生框架 — 完整测试套件

覆盖维度：
1. 解析器：变量提取、语法校验、AST 安全校验
2. 沙箱安全：注入攻击拦截、除以零处理
3. Pandas 引擎：正常计算、除以零、NaN 传播、缺失值预填充
4. Online 引擎：正常计算、除以零、None 传播、安全防护
5. 统一入口：自动引擎识别、异常冒泡
6. PySpark 引擎：标记为 skip（需 Spark 环境）

运行方式：
    pytest tests/test_feature_derivative.py -v
"""

import pytest
import numpy as np
import pandas as pd

from feature_derivative import (
    transform,
    MissingVariableError,
    ExpressionSyntaxError,
    UnsafeExpressionError,
)
from feature_derivative.parser import extract_variables, validate_ast_safety
from feature_derivative.sandbox import safe_eval_expression


# ============================================================
# 1. 解析器测试
# ============================================================
class TestExtractVariables:
    """变量提取测试"""

    def test_simple(self):
        """基本二元表达式"""
        assert set(extract_variables("a + b")) == {"a", "b"}

    def test_complex_expression(self):
        """带括号和重复变量的表达式"""
        assert set(extract_variables("a/(a+b)")) == {"a", "b"}

    def test_with_constants(self):
        """含数字常量的表达式"""
        assert set(extract_variables("a * 2 + b / 3")) == {"a", "b"}

    def test_nested_parentheses(self):
        """多层嵌套括号"""
        assert set(extract_variables("((a+b)*c)/(a-b)")) == {"a", "b", "c"}

    def test_single_variable(self):
        """单变量表达式"""
        assert extract_variables("a") == ["a"]

    def test_preserves_order(self):
        """保持变量首次出现的顺序"""
        result = extract_variables("c + a + b + a")
        assert result == ["c", "a", "b"]

    def test_negative_unary(self):
        """一元负号"""
        assert set(extract_variables("-a + b")) == {"a", "b"}

    def test_syntax_error(self):
        """语法错误应抛出 ExpressionSyntaxError"""
        with pytest.raises(ExpressionSyntaxError):
            extract_variables("a +")

    def test_empty_expression(self):
        """空表达式应抛出 ExpressionSyntaxError"""
        with pytest.raises(ExpressionSyntaxError):
            extract_variables("")


class TestValidateAstSafety:
    """AST 安全校验测试"""

    def test_safe_expression(self):
        """合法四则运算表达式应通过校验"""
        tree = validate_ast_safety("a/(a+b)")
        assert tree is not None

    def test_function_call_blocked(self):
        """函数调用应被拦截"""
        with pytest.raises(UnsafeExpressionError):
            validate_ast_safety("abs(a)")

    def test_attribute_access_blocked(self):
        """属性访问应被拦截"""
        with pytest.raises(UnsafeExpressionError):
            validate_ast_safety("a.__class__")

    def test_import_blocked(self):
        """__import__ 应被拦截"""
        with pytest.raises(UnsafeExpressionError):
            validate_ast_safety("__import__('os')")

    def test_subscript_blocked(self):
        """下标访问应被拦截"""
        with pytest.raises(UnsafeExpressionError):
            validate_ast_safety("a[0]")

    def test_boolean_op_blocked(self):
        """布尔运算应被拦截（不属于四则运算）"""
        with pytest.raises(UnsafeExpressionError):
            validate_ast_safety("a and b")


# ============================================================
# 2. 沙箱安全测试
# ============================================================
class TestSandbox:
    """安全沙箱测试"""

    def test_basic_addition(self):
        assert safe_eval_expression("a + b", {"a": 1, "b": 2}) == 3

    def test_division(self):
        assert safe_eval_expression("a / b", {"a": 10, "b": 2}) == 5.0

    def test_complex_expression(self):
        result = safe_eval_expression("a/(a+b)", {"a": 1, "b": 4})
        assert abs(result - 0.2) < 1e-10

    def test_division_by_zero(self):
        """除以零返回 None"""
        assert safe_eval_expression("a / b", {"a": 1, "b": 0}) is None

    def test_none_value(self):
        """None 参与运算返回 None"""
        assert safe_eval_expression("a + b", {"a": 1, "b": None}) is None

    def test_unsafe_import(self):
        """尝试导入模块应抛出异常"""
        with pytest.raises(UnsafeExpressionError):
            safe_eval_expression("__import__('os').system('rm -rf /')", {})

    def test_unsafe_open(self):
        """尝试打开文件应抛出异常"""
        with pytest.raises(UnsafeExpressionError):
            safe_eval_expression("open('/etc/passwd')", {})

    def test_unsafe_eval(self):
        """尝试嵌套 eval 应抛出异常"""
        with pytest.raises(UnsafeExpressionError):
            safe_eval_expression("eval('1+1')", {})

    def test_unsafe_builtins(self):
        """尝试访问 __builtins__ 应被拦截（AST 层面无 Name '__builtins__'，
        但即便绕过 AST，__builtins__ 为空也无法使用）"""
        # 直接使用 __builtins__ 作为变量名会被 Name 节点捕获，
        # 但不会匹配 AST 白名单外的操作，所以这里测试运行时安全性
        result = safe_eval_expression("a + 1", {"a": 1, "__builtins__": {}})
        # 即使调用方恶意传入 __builtins__，沙箱仍会将其覆盖为空
        assert result == 2

    def test_no_globals_access(self):
        """验证全局命名空间为空，无法访问 os/sys"""
        # AST 校验会先拦截 __import__，这里做双重验证
        with pytest.raises(UnsafeExpressionError):
            safe_eval_expression("__import__('os')", {})
        with pytest.raises(UnsafeExpressionError):
            safe_eval_expression("__import__('sys')", {})

    def test_negative_number(self):
        """一元负号"""
        assert safe_eval_expression("-a + b", {"a": 3, "b": 10}) == 7

    def test_parenthesized_expression(self):
        """括号嵌套"""
        assert safe_eval_expression("(a+b)*(a-b)", {"a": 5, "b": 3}) == 16


# ============================================================
# 3. Pandas 引擎测试
# ============================================================
class TestPandasStrategy:
    """Pandas 计算引擎测试"""

    def test_basic_calculation(self):
        """基本计算"""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = transform(df, "a/(a+b)", "new_feature")
        expected = [1 / 5, 2 / 7, 3 / 9]
        np.testing.assert_array_almost_equal(result["new_feature"].values, expected)

    def test_multiplication(self):
        """乘法"""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = transform(df, "a * b", "product")
        np.testing.assert_array_equal(result["product"].values, [4, 10, 18])

    def test_division_by_zero(self):
        """除以零 → inf → NaN"""
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [-1.0, -2.0]})
        result = transform(df, "a/(a+b)", "new_feature")
        # a+b = 0，除以零产生 inf，替换为 NaN
        assert result["new_feature"].isna().all()

    def test_nan_propagation(self):
        """NaN 传播：输入含 NaN → 输出 NaN"""
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
        result = transform(df, "a/(a+b)", "new_feature")
        assert pd.isna(result["new_feature"].iloc[1])
        assert not pd.isna(result["new_feature"].iloc[0])
        assert not pd.isna(result["new_feature"].iloc[2])

    def test_fill_value(self):
        """预填充缺失值"""
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
        result = transform(df, "a/(a+b)", "new_feature", fill_value=0)
        # NaN 被填充为 0：0/(0+5) = 0
        assert result["new_feature"].iloc[1] == 0.0

    def test_missing_variable(self):
        """变量缺失应抛出 MissingVariableError"""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(MissingVariableError) as exc_info:
            transform(df, "a/(a+b+c)", "new_feature")
        assert "c" in str(exc_info.value.missing_vars)

    def test_original_dataframe_unchanged(self):
        """原始 DataFrame 不应被修改"""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        original_cols = df.columns.tolist()
        transform(df, "a+b", "sum_col")
        assert df.columns.tolist() == original_cols

    def test_subtraction(self):
        """减法"""
        df = pd.DataFrame({"a": [10, 20, 30], "b": [1, 2, 3]})
        result = transform(df, "a - b", "diff")
        np.testing.assert_array_equal(result["diff"].values, [9, 18, 27])


# ============================================================
# 4. Online (Dict) 引擎测试
# ============================================================
class TestOnlineStrategy:
    """Python 单变量引擎测试"""

    def test_basic_calculation(self):
        """基本计算"""
        result = transform({"a": 1, "b": 4}, "a/(a+b)", "new_feature")
        assert abs(result["new_feature"] - 0.2) < 1e-10

    def test_division_by_zero(self):
        """除以零返回 None"""
        result = transform({"a": 1, "b": -1}, "a/(a+b)", "new_feature")
        assert result["new_feature"] is None

    def test_none_propagation(self):
        """None 参与运算返回 None"""
        result = transform({"a": 1, "b": None}, "a/(a+b)", "new_feature")
        assert result["new_feature"] is None

    def test_fill_value(self):
        """预填充 None 为 0"""
        result = transform(
            {"a": 1, "b": None}, "a/(a+b)", "new_feature", fill_value=0
        )
        # b=None 被填充为 0: 1/(1+0) = 1.0
        assert abs(result["new_feature"] - 1.0) < 1e-10

    def test_missing_variable(self):
        """变量缺失应抛出 MissingVariableError"""
        with pytest.raises(MissingVariableError) as exc_info:
            transform({"a": 1}, "a/(a+b)", "new_feature")
        assert "b" in str(exc_info.value.missing_vars)

    def test_original_dict_unchanged(self):
        """原始 dict 不应被修改"""
        data = {"a": 1, "b": 4}
        original_keys = set(data.keys())
        transform(data, "a+b", "sum_col")
        assert set(data.keys()) == original_keys

    def test_addition(self):
        """加法"""
        result = transform({"a": 3, "b": 7}, "a + b", "sum")
        assert result["sum"] == 10

    def test_multiplication(self):
        """乘法"""
        result = transform({"a": 3, "b": 7}, "a * b", "product")
        assert result["product"] == 21

    def test_complex_nested(self):
        """复杂嵌套表达式"""
        result = transform(
            {"a": 5, "b": 3, "c": 2}, "(a+b)*(a-b)/(c)", "complex"
        )
        # (5+3)*(5-3)/2 = 8*2/2 = 8.0
        assert abs(result["complex"] - 8.0) < 1e-10

    def test_unsafe_expression_blocked(self):
        """不安全表达式应被拦截"""
        with pytest.raises(UnsafeExpressionError):
            transform({"a": 1}, "__import__('os').system('ls')", "hack")


# ============================================================
# 5. 统一入口 — 自动引擎识别测试
# ============================================================
class TestAutoDetection:
    """自动引擎识别测试"""

    def test_dict_detection(self):
        """dict → OnlineStrategy"""
        result = transform({"a": 1, "b": 2}, "a+b", "sum")
        assert isinstance(result, dict)
        assert result["sum"] == 3

    def test_dataframe_detection(self):
        """pd.DataFrame → PandasStrategy"""
        df = pd.DataFrame({"a": [1], "b": [2]})
        result = transform(df, "a+b", "sum")
        assert isinstance(result, pd.DataFrame)

    def test_unsupported_type(self):
        """不支持的类型应抛出 TypeError"""
        with pytest.raises(TypeError):
            transform([1, 2, 3], "a+b", "sum")

    def test_unsupported_string(self):
        """字符串类型应抛出 TypeError"""
        with pytest.raises(TypeError):
            transform("a=1,b=2", "a+b", "sum")


# ============================================================
# 6. PySpark 引擎测试（需 Spark 环境，默认 skip）
# ============================================================
class TestSparkStrategy:
    """PySpark 计算引擎测试 — 需要安装 PySpark"""

    SPARK_AVAILABLE = False

    @classmethod
    def _check_spark(cls):
        try:
            from pyspark.sql import SparkSession

            cls.SPARK_AVAILABLE = True
        except ImportError:
            cls.SPARK_AVAILABLE = False

    def _get_spark(self):
        from pyspark.sql import SparkSession

        return (
            SparkSession.builder.master("local[1]")
            .appName("feature_derivative_test")
            .config("spark.sql.ansi.enabled", "false")
            .getOrCreate()
        )

    @pytest.fixture(autouse=True)
    def skip_without_spark(self):
        self._check_spark()
        if not self.SPARK_AVAILABLE:
            pytest.skip("PySpark 未安装，跳过 Spark 引擎测试")

    def test_basic_calculation(self):
        """基本计算"""
        spark = self._get_spark()
        try:
            sdf = spark.createDataFrame([(1, 4), (2, 5), (3, 6)], ["a", "b"])
            result = transform(sdf, "a/(a+b)", "new_feature")
            rows = result.orderBy("a").collect()
            assert abs(rows[0]["new_feature"] - 1 / 5) < 1e-6
            assert abs(rows[1]["new_feature"] - 2 / 7) < 1e-6
            assert abs(rows[2]["new_feature"] - 3 / 9) < 1e-6
        finally:
            spark.stop()

    def test_division_by_zero(self):
        """除以零 → null"""
        spark = self._get_spark()
        try:
            sdf = spark.createDataFrame([(1, -1)], ["a", "b"])
            result = transform(sdf, "a/(a+b)", "new_feature")
            row = result.collect()[0]
            assert row["new_feature"] is None
        finally:
            spark.stop()

    def test_null_propagation(self):
        """null 传播"""
        spark = self._get_spark()
        try:
            sdf = spark.createDataFrame([(1, None)], ["a", "b"])
            result = transform(sdf, "a/(a+b)", "new_feature")
            row = result.collect()[0]
            assert row["new_feature"] is None
        finally:
            spark.stop()

    def test_missing_variable(self):
        """变量缺失"""
        spark = self._get_spark()
        try:
            sdf = spark.createDataFrame([(1, 2)], ["a", "b"])
            with pytest.raises(MissingVariableError):
                transform(sdf, "a/(a+b+c)", "new_feature")
        finally:
            spark.stop()
