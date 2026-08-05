"""spark.py 纯函数测试（无需 pyspark）

覆盖：mapInPandas 处理函数输出列=原列+risk_score 且打分一致 /
      exec 生成固定 arity 标量函数入参一致 / 缺失特征列明确报错
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from online_deploy_proto import build_engine
from online_deploy_proto import deploy_spec_pb2 as pb
from online_deploy_proto.serialize import to_proto_bytes
from online_deploy_proto.spark import (
    _make_map_in_pandas_fn,
    _make_scalar_fn,
    _score_pdf,
)


def _minimal_spec(features, code="def score(input):\n    return 0.5"):
    """单特征/单算子最小 spec（常数模型），用于 arity 边界测试。"""
    s = pb.DeploySpec()
    s.version = 1
    s.feature_names_in.extend(features)
    m = s.model.m2cgen
    m.feature_names.extend(features)
    m.base_score = 0.5
    m.code = code
    return s.SerializeToString()


class TestMapInPandas:
    def test_score_pdf_columns(self, deploy, spec_bytes, trained):
        _, X, _ = trained
        scorer = build_engine(spec_bytes)
        pdf = X.iloc[:10]
        out = _score_pdf(pdf, scorer, scorer.feature_names_in)
        # 输出列 = 原列 + risk_score（mapInPandas schema 完全一致要求）
        assert list(out.columns) == list(pdf.columns) + ["risk_score"]
        truth = deploy.score_batch(pdf.to_dict("records"))
        assert np.abs(out["risk_score"].to_numpy() - truth).max() < 1e-9

    def test_map_in_pandas_fn(self, spec_bytes, trained):
        _, X, _ = trained
        scorer = build_engine(spec_bytes)
        fn = _make_map_in_pandas_fn(SimpleNamespace(value=spec_bytes))
        pdfs = [X.iloc[:5], X.iloc[5:12]]
        out = pd.concat(list(fn(iter(pdfs))), ignore_index=True)
        assert list(out.columns) == list(X.columns) + ["risk_score"]
        truth = scorer.score_rows(X.iloc[:12].to_dict("records"))
        assert np.abs(out["risk_score"].to_numpy() - truth).max() < 1e-9

    def test_non_numeric_cell_coerced(self, spec_bytes, trained):
        """非数值 cell → NaN（pd.to_numeric coerce，与 _to_array 语义一致）。"""
        _, X, _ = trained
        scorer = build_engine(spec_bytes)
        pdf = X.iloc[:3].copy().astype({X.columns[0]: "object"})
        pdf.iloc[0, 0] = "abc"
        out = _score_pdf(pdf, scorer, scorer.feature_names_in)
        assert out["risk_score"].notna().all()

    def test_missing_feature_raises(self, spec_bytes, trained):
        _, X, _ = trained
        scorer = build_engine(spec_bytes)
        pdf = X.iloc[:3].drop(columns=[X.columns[0]])
        with pytest.raises(KeyError):
            _score_pdf(pdf, scorer, scorer.feature_names_in)


class TestScalarFn:
    def test_fixed_arity_matches(self, spec_bytes, trained):
        _, X, _ = trained
        scorer = build_engine(spec_bytes)
        feats = scorer.feature_names_in
        f = _make_scalar_fn(spec_bytes, feats)
        out = f(*[X[c] for c in feats])
        truth = scorer.score_rows(X.to_dict("records"))
        assert np.abs(out.to_numpy() - truth).max() < 1e-9

    def test_single_feature_arity(self):
        spec = _minimal_spec(["a"])
        f = _make_scalar_fn(spec, ["a"])
        out = f(pd.Series([1.0, 2.0, np.nan]))
        assert (out.to_numpy() == 0.5).all()

    def test_empty_features_rejected(self, spec_bytes):
        with pytest.raises(ValueError):
            _make_scalar_fn(spec_bytes, [])
