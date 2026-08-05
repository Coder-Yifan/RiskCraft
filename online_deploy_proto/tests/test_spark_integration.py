"""PySpark 集成测试（本机未装 pyspark / 3.2.4 与 Python 3.12 不兼容 → skip）

在具备 pyspark 3.2.4 + Python ≤3.10 的环境下运行：
    spark-submit --py-files online_deploy_proto/ tests/test_spark_integration.py
并开启 Arrow：spark.sql.execution.arrow.pyspark.enabled=true
"""

import numpy as np
import pytest

pytest.importorskip("pyspark")

from pyspark.sql import SparkSession

from online_deploy_proto.spark import add_risk_score, make_scalar_pandas_udf


@pytest.fixture(scope="module")
def spark():
    s = SparkSession.builder.master("local[2]").appName("riskcraft-test").getOrCreate()
    s.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    yield s
    s.stop()


def test_map_in_pandas_scoring(spark, deploy, spec_bytes, trained):
    from pyspark.sql.functions import col

    _, X, _ = trained
    df = spark.createDataFrame(X.iloc[:50])
    out = add_risk_score(df, spec_bytes).collect()
    scores = [r["risk_score"] for r in out]
    truth = deploy.score_batch(X.iloc[:50].to_dict("records"))
    assert "risk_score" in out[0].asDict()
    assert np.abs(np.array(scores) - truth).max() < 1e-6


def test_scalar_udf_scoring(spark, deploy, spec_bytes, trained):
    from pyspark.sql.functions import col

    _, X, _ = trained
    df = spark.createDataFrame(X.iloc[:20])
    fnames = deploy.feature_names_in_
    udf = make_scalar_pandas_udf(spec_bytes, fnames)
    out = df.select(udf(*[col(c) for c in fnames]).alias("risk_score")).collect()
    scores = [r["risk_score"] for r in out]
    truth = deploy.score_batch(X.iloc[:20].to_dict("records"))
    assert np.abs(np.array(scores) - truth).max() < 1e-6
