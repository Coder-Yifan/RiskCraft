"""
特征衍生框架 — 完整演示

分别演示在 Pandas、PySpark 和 Dict 三种输入下，
计算表达式 "a/(a+b)" 并生成新列 "new_feature" 的完整流程。

运行方式：
    python demo.py
"""

import sys
import io

# Windows GBK 终端兼容：强制 UTF-8 输出
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from feature_derivative import transform, MissingVariableError, UnsafeExpressionError


def demo_pandas():
    """Pandas 引擎演示"""
    print("=" * 60)
    print("[Pandas 引擎演示]")
    print("=" * 60)

    # 基本计算
    print("\n[1] 基本计算: a/(a+b)")
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0],'c':[2,4,6]})
    result = transform(df, "(a+c)/(a+b)", "new_feature")
    print(result.to_string(index=False))

    # 除以零
    print("\n[2] 除以零处理: a+b=0 时")
    df_zero = pd.DataFrame({"a": [1.0, 2.0], "b": [-1.0, -2.0]})
    result_zero = transform(df_zero, "a/(a+b)", "new_feature")
    print(result_zero.to_string(index=False))
    print("   -> inf 被替换为 NaN")

    # NaN 传播
    print("\n[3] NaN 传播: 输入含 NaN")
    df_nan = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
    result_nan = transform(df_nan, "a/(a+b)", "new_feature")
    print(result_nan.to_string(index=False))
    print("   -> NaN 输入 -> NaN 输出 (传播策略)")

    # 预填充
    print("\n[4] 预填充模式: fill_value=0")
    result_fill = transform(df_nan, "a/(a+b)", "new_feature", fill_value=0)
    print(result_fill.to_string(index=False))
    print("   -> NaN 被预填充为 0 后计算")


def demo_pyspark():
    """PySpark 引擎演示"""
    print("\n" + "=" * 60)
    print("[PySpark 引擎演示]")
    print("=" * 60)

    try:
        from pyspark.sql import SparkSession

        spark = (
            SparkSession.builder.master("local[1]")
            .appName("feature_derivative_demo")
            .config("spark.sql.ansi.enabled", "false")
            .getOrCreate()
        )

        # 基本计算
        print("\n[1] 基本计算: a/(a+b)")
        sdf = spark.createDataFrame([(1.0, 4.0), (2.0, 5.0), (3.0, 6.0)], ["a", "b"])
        result = transform(sdf, "a/(a+b)", "new_feature")
        result.orderBy("a").show()

        # 除以零
        print("[2] 除以零处理: a+b=0 时")
        sdf_zero = spark.createDataFrame([(1.0, -1.0), (2.0, -2.0)], ["a", "b"])
        result_zero = transform(sdf_zero, "a/(a+b)", "new_feature")
        result_zero.show()
        print("   -> 除以零返回 null")

        spark.stop()

    except ImportError:
        print("\n[!] PySpark 未安装，跳过 Spark 演示")
        print("    安装方式: pip install pyspark")


def demo_online():
    """Online (Dict) 引擎演示"""
    print("\n" + "=" * 60)
    print("[Online (Dict) 引擎演示]")
    print("=" * 60)

    # 基本计算
    print("\n[1] 基本计算: a/(a+b)")
    result = transform({"a": 1, "b": 4}, "a/(a+b)", "new_feature")
    print(f"   输入: a=1, b=4 -> new_feature={result['new_feature']}")

    # 除以零
    print("\n[2] 除以零处理: a+b=0")
    result_zero = transform({"a": 1, "b": -1}, "a/(a+b)", "new_feature")
    print(f"   输入: a=1, b=-1 -> new_feature={result_zero['new_feature']}")
    print("   -> 除以零返回 None")

    # None 传播
    print("\n[3] None 传播: 输入含 None")
    result_none = transform({"a": 1, "b": None}, "a/(a+b)", "new_feature")
    print(f"   输入: a=1, b=None -> new_feature={result_none['new_feature']}")
    print("   -> None 参与运算返回 None")

    # 预填充
    print("\n[4] 预填充模式: fill_value=0")
    result_fill = transform(
        {"a": 1, "b": None}, "a/(a+b)", "new_feature", fill_value=0
    )
    print(f"   输入: a=1, b=None(fill=0) -> new_feature={result_fill['new_feature']}")
    print("   -> None 被预填充为 0: 1/(1+0)=1.0")


def demo_error_handling():
    """异常处理演示"""
    print("\n" + "=" * 60)
    print("[异常处理演示]")
    print("=" * 60)

    # 变量缺失
    print("\n[1] 变量缺失")
    try:
        transform({"a": 1}, "a/(a+b)", "new_feature")
    except MissingVariableError as e:
        print(f"   MissingVariableError: {e}")

    # 不安全表达式
    print("\n[2] 不安全表达式 (注入攻击)")
    try:
        transform({"a": 1}, "__import__('os').system('ls')", "hack")
    except UnsafeExpressionError as e:
        print(f"   UnsafeExpressionError: {e}")

    # 不支持的数据类型
    print("\n[3] 不支持的数据类型")
    try:
        transform([1, 2, 3], "a+b", "sum")
    except TypeError as e:
        print(f"   TypeError: {e}")


def demo_complex_expressions():
    """复杂表达式演示"""
    print("\n" + "=" * 60)
    print("[复杂表达式演示]")
    print("=" * 60)

    expressions = [
        ("a*b + c/d", {"a": 2, "b": 3, "c": 10, "d": 5}),
        ("((a+b)*c)/(a-b)", {"a": 5, "b": 3, "c": 2}),
        ("-a + b * c", {"a": 1, "b": 3, "c": 4}),
    ]

    for expr, data in expressions:
        result = transform(data, expr, "result")
        print(f"\n   表达式: {expr}")
        print(f"   输入: {data}")
        print(f"   结果: {result['result']}")


if __name__ == "__main__":
    demo_pandas()
    # demo_pyspark()
    # demo_online()
    # demo_error_handling()
    # demo_complex_expressions()

    # print("\n" + "=" * 60)
    # print("[OK] 所有演示完成!")
    # print("=" * 60)
