"""
一致性校验 — assert_consistent

以 sklearn pipeline.predict_proba 为离线真值，对部署流水线打分做偏差校验。
标准：宽松 atol=1e-4（用户确认）。实测 ONNX/m2cgen 后端偏差 ~1e-7，富余百倍。

测试样本覆盖：
- 随机样本（真实分布）
- 分箱边界邻域（±eps，测 float32 临界一致性）
- 缺失行 / 哨兵行（测填充一致性）
"""

import numpy as np
import pandas as pd

from .exceptions import ConsistencyError


def _base_row(X, index=0):
    """取一行真实样本作为边界/缺失样本的基座。"""
    return X.iloc[index].to_dict()


def _boundary_rows(pipe, base, eps=1e-7, max_edges=50):
    """从 pipeline 分箱边界生成临界邻域样本。"""
    rows = []
    for _, step in getattr(pipe, "steps", []):
        edges_map = getattr(step, "bin_edges_", None)
        if not edges_map:
            continue
        for col, edges in edges_map.items():
            if col not in base or len(edges) < 3:
                continue
            for e in edges[1:-1][:max_edges]:
                for delta in (-eps, 0.0, eps):
                    r = dict(base)
                    r[col] = float(e) + delta
                    rows.append(r)
    return rows


def generate_test_rows(pipe, X, n_random=200, n_missing=3, n_sentinel=3,
                       boundary_eps=1e-7):
    """生成一致性校验样本集（list[dict]）。

    Args:
        pipe: 已拟合 pipeline（用于提取分箱边界）
        X: 训练/测试 DataFrame（随机抽样来源）
        n_random: 随机样本数
        n_missing: 缺失样本数（随机列置 NaN）
        n_sentinel: 哨兵样本数（随机列置 -999）
        boundary_eps: 分箱边界邻域扰动幅度
    """
    if len(X) == 0:
        raise ValueError("X 为空，无法生成测试样本")
    rows = []

    # 1. 随机样本
    n_smp = min(n_random, len(X))
    sample = X.sample(n=n_smp, random_state=0)
    rows.extend(sample.to_dict("records"))

    base = _base_row(X, index=0)

    # 2. 缺失样本：随机列置 NaN
    cols = list(X.columns)
    for _ in range(n_missing):
        r = dict(base)
        for c in np.random.choice(cols, size=min(3, len(cols)), replace=False):
            r[c] = np.nan
        rows.append(r)

    # 3. 哨兵样本：随机列置 -999
    for _ in range(n_sentinel):
        r = dict(base)
        for c in np.random.choice(cols, size=min(2, len(cols)), replace=False):
            r[c] = -999
        rows.append(r)

    # 4. 分箱边界邻域样本
    rows.extend(_boundary_rows(pipe, base, eps=boundary_eps))

    return rows


def assert_consistent(pipe, deploy, X=None, atol=1e-4, n_random=200, on_fail=None):
    """校验部署流水线与 sklearn pipeline 打分一致性。

    Args:
        pipe: 已拟合的 sklearn pipeline（真值来源）
        deploy: DeployPipeline
        X: 样本数据（None 时仅用边界/缺失样本，不抽样）
        atol: 绝对偏差阈值（默认 1e-4）
        n_random: 随机样本数
        on_fail: 失败回调（默认抛 ConsistencyError）

    Returns:
        dict: {"max_diff", "n_checked", "n_fail"}

    Raises:
        ConsistencyError: 存在偏差超过 atol 的样本
    """
    if X is not None:
        rows = generate_test_rows(pipe, X, n_random=n_random)
    else:
        base = {}
        for _, step in getattr(pipe, "steps", []):
            if hasattr(step, "feature_names_in_"):
                base = {c: np.nan for c in step.feature_names_in_}
        rows = generate_test_rows(pipe, pd.DataFrame([base]), n_random=0, n_missing=0,
                                  n_sentinel=0)
        rows = rows + _boundary_rows(pipe, base)
    if not rows:
        raise ConsistencyError("未生成任何测试样本")

    columns = deploy.feature_names_in_
    X_true = pd.DataFrame(rows, columns=columns)
    if getattr(pipe, "score_scaler", None) is not None:
        # 拉伸 pipeline：真值 = 风险分（分数域，容差放宽到 1.0）
        y_true = pipe.predict_score(X_true)
        atol = max(atol, 1.0)
    else:
        # 无拉伸：真值 = 概率，与原逻辑逐值一致（atol=1e-4 不变）
        y_true = pipe.predict_proba(X_true)[:, 1]
    y_deploy = deploy.score_batch(rows)

    diff = np.abs(y_true - y_deploy)
    n_fail = int((diff > atol).sum())
    result = {
        "max_diff": float(diff.max()) if len(diff) else 0.0,
        "n_checked": len(rows),
        "n_fail": n_fail,
    }

    if n_fail > 0:
        msg = (f"一致性校验失败: {n_fail}/{len(rows)} 行偏差超过 atol={atol}，"
               f"最大偏差 {float(diff.max()):.2e}")
        if on_fail is not None:
            on_fail(result, msg)
        else:
            raise ConsistencyError(msg)
    return result
