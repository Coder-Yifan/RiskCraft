"""
端到端演示：训练 pipeline → 编译 → proto 序列化 → 还原/自包含打分 → 字节对比

用法：
    D:/softwares/conda/python.exe online_deploy_proto/demo_proto.py
"""

import numpy as np
import pandas as pd

from risk_ml import RiskPipeline
from risk_ml.encoding import BinnerWoeEncoder
from risk_ml.estimator import RiskXGBClassifier
from risk_ml.feature_selection import CorrelationSelector, IVSelector
from risk_ml.preprocessing import FeatureCleaner
from risk_ml.online_deploy import PipelineParser, assert_consistent
from risk_ml.online_deploy.demo_deploy import make_data

from online_deploy_proto import build_engine
from online_deploy_proto.serialize import from_proto_bytes, to_proto_bytes


def main():
    print("=" * 60)
    print("online_deploy_proto demo")
    print("=" * 60)

    # ---- 1. 数据 + 训练 ----
    df = make_data(n=2000, seed=42)
    X = df.drop(columns=["y"])
    y = df["y"]

    pipe = RiskPipeline([
        ("cleaner", FeatureCleaner(sentinels=[-999])),
        ("binner_woe", BinnerWoeEncoder(max_bins=6)),
        ("iv_selector", IVSelector(iv_threshold=0.02)),
        ("corr_selector", CorrelationSelector(corr_threshold=0.8)),
        ("xgb", RiskXGBClassifier(n_estimators=50, max_depth=4, eval_metric="auc")),
    ])
    pipe.fit(X, y)

    # ---- 2. 编译 + proto 序列化（双后端）----
    print("\n[proto 序列化]")
    for backend in ["m2cgen", "onnx"]:
        deploy = PipelineParser(backend=backend).compile_pipeline(pipe)
        spec = to_proto_bytes(deploy)
        j = deploy.to_json().encode("utf-8")

        # 还原为 DeployPipeline，校验与 sklearn 一致
        back = from_proto_bytes(spec)
        r = assert_consistent(pipe, back, X=X, atol=1e-4)

        # 自包含打分器（executor 侧语义，无 risk_ml 依赖路径）
        scorer = build_engine(spec)
        row = X.iloc[0].to_dict()
        row["amount"] = 3000.0
        p_deploy = back.score(row)
        p_scorer = scorer.score(row)
        p_truth = pipe.predict_proba(pd.DataFrame([row]))[0, 1]

        print(f"  [{backend}] proto {len(spec):>6} B vs JSON {len(j):>6} B "
              f"({len(spec) / len(j) * 100:.0f}%) | "
              f"一致性 max_diff={r['max_diff']:.2e} fail={r['n_fail']}")
        print(f"          单条打分: deploy={p_deploy:.6f} scorer={p_scorer:.6f} "
              f"sklearn={p_truth:.6f}")

    # ---- 3. 批量打分 + executor 内核（无 risk_ml 语义）----
    deploy = PipelineParser(backend="m2cgen").compile_pipeline(pipe)
    spec = to_proto_bytes(deploy)
    scorer = build_engine(spec)
    rows = X.iloc[:200].to_dict("records")
    p = scorer.score_rows(rows)
    print(f"\n[executor 批量打分] 200 条: 形状 {p.shape}, 范围 [{p.min():.4f}, {p.max():.4f}]")
    assert np.abs(p - deploy.score_batch(rows)).max() < 1e-9
    print("[PASS] executor 与 DeployPipeline 逐位一致 (max_diff<1e-9)")

    # ---- 4. 字节确定性（可作缓存 key）----
    print(f"\n[字节确定性] to_proto_bytes 两次相等: {to_proto_bytes(deploy) == spec}")
    print("\nDone.")


if __name__ == "__main__":
    main()
