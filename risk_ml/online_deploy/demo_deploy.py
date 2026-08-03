"""
端到端演示：训练 pipeline → 编译部署 → 一致性校验 → 性能基准 → 序列化

用法：
    D:/softwares/conda/python.exe risk_ml/online_deploy/demo_deploy.py
"""

import numpy as np
import pandas as pd

from risk_ml import RiskPipeline
from risk_ml.encoding import BinnerWoeEncoder
from risk_ml.estimator import RiskXGBClassifier
from risk_ml.feature_selection import CorrelationSelector, IVSelector
from risk_ml.preprocessing import FeatureCleaner

from . import assert_consistent, benchmark
from .parser import PipelineParser


def make_data(n=5000, seed=42):
    """合成风控数据（纯数值特征）。"""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "amount": np.random.lognormal(mean=6.0, sigma=0.8, size=n),
        "age": np.clip(np.random.normal(35, 8, n), 18, 65),
        "income": np.random.lognormal(mean=8.0, sigma=0.7, size=n),
        "freq": np.random.poisson(3, n).astype(float),
        "noise1": rng.normal(size=n),
        "noise2": rng.normal(size=n),
    })
    # 5% 缺失（模拟真实风控数据）
    for col in ["amount", "income", "age"]:
        mask = rng.random(n) < 0.05
        df.loc[mask, col] = -999  # 用哨兵值表示缺失
    logit = 0.5 * np.log(df["amount"].clip(lower=1)) + 0.02 * df["age"] - 2.2
    prob = 1 / (1 + np.exp(-logit))
    df["y"] = (rng.random(n) < prob).astype(int)
    return df


def main():
    print("=" * 60)
    print("RiskCraft online_deploy demo")
    print("=" * 60)

    # ---- 1. 数据 + 训练 ----
    df = make_data()
    X = df.drop(columns=["y"])
    y = df["y"]
    print(f"数据: {len(df)} 行 x {X.shape[1]} 特征, 正例率 {y.mean():.3f}")

    pipe = RiskPipeline([
        ("cleaner", FeatureCleaner(sentinels=[-999])),
        ("binner_woe", BinnerWoeEncoder(max_bins=6)),
        ("iv_selector", IVSelector(iv_threshold=0.02)),
        ("corr_selector", CorrelationSelector(corr_threshold=0.8)),
        ("xgb", RiskXGBClassifier(n_estimators=100, max_depth=4, eval_metric="auc")),
    ])
    pipe.fit(X, y)
    print(f"pipeline 保留特征: {pipe[:-1].transform(X).shape[1]} 个")

    # ---- 2. 编译双后端 ----
    deploy_m2c = PipelineParser(backend="m2cgen").compile_pipeline(pipe)
    deploy_onnx = PipelineParser(backend="onnx").compile_pipeline(pipe)
    print("\n[m2cgen backend]")
    print(deploy_m2c.describe())
    print("[onnx backend]")
    print(deploy_onnx.describe())

    # ---- 3. 一致性校验 ----
    for name, deploy in [("m2cgen", deploy_m2c), ("onnx", deploy_onnx)]:
        result = assert_consistent(pipe, deploy, X=X, atol=1e-4)
        print(f"\n一致性({name}): max_diff={result['max_diff']:.2e} "
              f"checked={result['n_checked']} fail={result['n_fail']}  [PASS]")

    # ---- 4. 单条 + 批量打分 ----
    row = X.iloc[0].to_dict()
    row["amount"] = 3000.0
    print(f"\n单条打分: score({row})")
    print(f"  m2cgen: {deploy_m2c.score(row):.6f}")
    print(f"  onnx:   {deploy_onnx.score(row):.6f}")
    print(f"  真值:   {pipe.predict_proba(pd.DataFrame([row]))[0, 1]:.6f}")

    batch = X.iloc[:100].to_dict("records")
    p_batch = deploy_m2c.score_batch(batch)
    print(f"批量打分(100 条): 形状 {p_batch.shape}, 范围 [{p_batch.min():.4f}, {p_batch.max():.4f}]")

    # ---- 5. 性能基准 ----
    print("\n[性能基准]")
    for name, deploy in [("m2cgen", deploy_m2c), ("onnx", deploy_onnx)]:
        b = benchmark(deploy, X)
        print(f"  {name:7s}: 单条 {b['single_us']:7.2f} us | "
              f"批量{b['batch_size']} {b['batch_ms']:6.2f} ms | QPS {b['qps']:.0f}")

    # ---- 6. 序列化 round-trip ----
    from .parser import DeployPipeline
    s = deploy_m2c.to_json()
    deploy_back = DeployPipeline.from_json(s)
    result = assert_consistent(pipe, deploy_back, X=X, atol=1e-4)
    print(f"\n序列化 round-trip: JSON {len(s)/1024:.1f} KB, "
          f"反序列化后一致性 max_diff={result['max_diff']:.2e}  [PASS]")
    print("\nDone.")


if __name__ == "__main__":
    main()
