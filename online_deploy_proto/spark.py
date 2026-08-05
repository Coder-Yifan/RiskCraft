"""
PySpark 3.2.4 批量打分

两种方式（pyspark 均惰性 import，无 pyspark 时纯函数可测）：
- add_risk_score: DataFrame → 追加 risk_score 列，用 mapInPandas（3.0+）。
  返回 schema 必须与输入列完全一致：fn 内 pdf.copy() 再追加列，不能只返回分数列。
- make_scalar_pandas_udf: exec 生成**固定 arity** 标量 pandas_udf。
  3.2.4 下标量 pandas_udf 的 *cols 变参签名不可靠且非文档化，故按特征数
  exec 生成 (c0, c1, ..., cn) 固定签名。

Spark 3.2.4 注意：DataFrame 级 scalar pandas_udf（入参为整个 DataFrame）是 3.4+，
本模块只用 mapInPandas（DataFrame 批）与逐列 Series 标量 UDF（3.0+ 均支持）。

非数值 cell 语义：入口 pd.to_numeric(errors="coerce") → NaN，与
DeployPipeline._to_array 的「非数值→NaN」一致。
"""

import numpy as np

from .scorer import build_engine


# ======================================================================
# mapInPandas（DataFrame 批）
# ======================================================================
def _score_pdf(pdf, scorer, feature_names):
    """单批 DataFrame → 原列 + risk_score。非数值 cell → NaN。"""
    import pandas as pd

    # 按特征名选列（与模型训练列序一致，与 DataFrame 列序解耦）
    sub = pdf[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    scores = scorer.score_np(sub)
    out = pdf.copy()  # 必须保留全部原列（mapInPandas schema 完全一致要求）
    out["risk_score"] = scores
    return out


def _make_map_in_pandas_fn(bc):
    """返回 mapInPandas 处理函数：迭代器[DataFrame] → 迭代器[DataFrame]。

    bc 为 Broadcast 对象（含 .value），测试可用 SimpleNamespace(value=...) 替代。
    """
    def fn(iter_of_pdf):
        scorer = build_engine(bc.value)
        feature_names = scorer.feature_names_in
        for pdf in iter_of_pdf:
            yield _score_pdf(pdf, scorer, feature_names)

    return fn


def add_risk_score(df, spec_bytes):
    """Spark DataFrame → 追加 risk_score 列（DoubleType）。

    Args:
        df: pyspark.sql.DataFrame（须含 spec 内 feature_names_in 全部列）
        spec_bytes: to_proto_bytes 产物（driver 侧广播到 executor）

    Returns:
        新 DataFrame：原列 + risk_score
    """
    from pyspark.sql import DataFrame
    from pyspark.sql.types import DoubleType, StructField

    if not isinstance(df, DataFrame):
        raise TypeError(f"df 必须是 pyspark.sql.DataFrame，收到 {type(df).__name__}")
    bc = df.sparkSession.sparkContext.broadcast(spec_bytes)
    schema = df.schema.add(StructField("risk_score", DoubleType(), True))
    return df.mapInPandas(_make_map_in_pandas_fn(bc), schema=schema)


# ======================================================================
# 标量 pandas_udf（逐列 Series）
# ======================================================================
def _make_scalar_fn(spec_bytes, feature_names):
    """exec 生成固定 arity 标量打分函数 (c0,...,cn) -> pd.Series。pyspark 无关。"""
    n = len(feature_names)
    if n == 0:
        raise ValueError("feature_names 不能为空")
    cols_params = ", ".join(f"c{i}" for i in range(n))
    cols_tuple = cols_params if n > 1 else cols_params + ","
    code = (
        "import numpy as np\n"
        "import pandas as pd\n"
        f"def _f({cols_params}):\n"
        "    sc = build_engine(spec_bytes)\n"
        f"    X = np.column_stack([np.asarray(c, dtype=np.float64) for c in ({cols_tuple})])\n"
        "    return pd.Series(sc.score_np(X))\n"
    )
    ns = {"build_engine": build_engine, "spec_bytes": spec_bytes}
    exec(compile(code, "<spark_scalar_udf>", "exec"), ns)
    return ns["_f"]


def make_scalar_pandas_udf(spec_bytes, feature_names=None):
    """生成固定 arity 标量 pandas_udf（返回 DoubleType 单列）。

    feature_names 缺省取 spec 内 feature_names_in。
    """
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import DoubleType

    if feature_names is None:
        feature_names = list(build_engine(spec_bytes).feature_names_in)
    return pandas_udf(
        _make_scalar_fn(spec_bytes, list(feature_names)), returnType=DoubleType()
    )
