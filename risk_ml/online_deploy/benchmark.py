"""
性能基准 — benchmark

测量部署流水线的单条延迟（线上实时打分核心指标）、批量延迟与 QPS。
"""

import time

import numpy as np
import pandas as pd


def _rows_from_frame(X, n):
    """从 DataFrame 取 n 行 dict 样本。"""
    if len(X) == 0:
        raise ValueError("X 为空")
    rows = []
    for i in range(n):
        rows.append(X.iloc[i % len(X)].to_dict())
    return rows


def benchmark(deploy, X, n_single=2000, warmup_single=300, batch_size=1000,
              n_batch=100, warmup_batch=10):
    """测量部署流水线性能。

    Args:
        deploy: DeployPipeline
        X: 样本 DataFrame
        n_single: 单条测量次数
        warmup_single: 单条预热次数
        batch_size: 批量打分样本数
        n_batch: 批量测量轮数
        warmup_batch: 批量预热轮数

    Returns:
        dict:
            single_us: 单条延迟（微秒/条）
            batch_ms: 批量延迟（毫秒/批）
            qps: 单条场景每秒可打分次数
    """
    rows = _rows_from_frame(X, n_single + warmup_single)

    # ---- 单条延迟 ----
    for r in rows[:warmup_single]:
        deploy.score(r)
    t0 = time.perf_counter()
    for r in rows[warmup_single:]:
        deploy.score(r)
    elapsed_single = time.perf_counter() - t0
    single_us = elapsed_single / n_single * 1e6

    # ---- 批量延迟 ----
    batch = rows[:batch_size]
    for _ in range(warmup_batch):
        deploy.score_batch(batch)
    t0 = time.perf_counter()
    for _ in range(n_batch):
        deploy.score_batch(batch)
    elapsed_batch = time.perf_counter() - t0
    batch_ms = elapsed_batch / n_batch * 1e3

    return {
        "single_us": round(single_us, 2),
        "batch_ms": round(batch_ms, 2),
        "qps": round(1e6 / single_us, 0),
        "batch_size": batch_size,
        "n_trees": getattr(getattr(deploy.model_op, "model_", None), "n_trees", None),
    }
