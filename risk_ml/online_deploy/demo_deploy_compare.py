"""
双 pipeline 在线部署对比 demo（xgboost / lightgbm）

对每个 pipeline：
1. 训练 sklearn RiskPipeline → PipelineParser 编译 DeployPipeline（m2cgen / onnx 双后端）
2. 生成部署文件：proto+m2cgen、proto+onnx 各一份，另存 pkl 参考模型
3. 一致性：以 pkl（pickle 反序列化 sklearn pipeline）预测为准，校验 proto+m2cgen / proto+onnx
4. 性能：单条打分迭代计时，对比 pkl / proto+m2cgen / proto+onnx

用法：
    D:/softwares/conda/python.exe risk_ml/online_deploy/demo_deploy_compare.py
"""

import os
import pickle
import tempfile
import time

import numpy as np
import pandas as pd

from risk_ml import RiskPipeline
from risk_ml.encoding import BinnerWoeEncoder
from risk_ml.estimator import RiskLGBMClassifier, RiskXGBClassifier
from risk_ml.feature_selection import IVSelector
from risk_ml.online_deploy import PipelineParser
from risk_ml.online_deploy.demo_deploy import make_data
from risk_ml.preprocessing import FeatureCleaner, FeatureDerivativeTransformer
from online_deploy_proto import build_engine
from online_deploy_proto.serialize import to_proto_bytes

ATOL = 1e-4        # 一致性阈值（与 test_deploy 一致）
N_CHECK = 500      # 一致性抽样行数
N_ITERS = 3000     # 部署后端单条打分迭代次数（快，多跑）
N_PKL_ITERS = 100  # pkl 单条打分迭代次数（慢，少跑）
SINGLE_ROW_IDX = 3  # 性能用单条行下标
OUT_DIR = os.path.join(tempfile.gettempdir(), "riskcraft_deploy_compare")


def _build_pipe(model_cls, X, y):
    """标准风控 pipeline：清洗 → 分箱WOE → IV筛选 → 树模型估计器。"""
    return RiskPipeline([
        ("cleaner", FeatureCleaner(sentinels=[-999])),
        ("bin_woe", BinnerWoeEncoder(max_bins=6)),
        ("select", IVSelector(iv_threshold=0.02)),
        ("model", model_cls(n_estimators=100, max_depth=4)),
    ]).fit(X, y)


def _build_pipe_fd(model_cls, X, y):
    """含 feature_derivative 的 pipeline：特征衍生 → 清洗 → 分箱WOE → 筛选 → 模型。"""
    return RiskPipeline([
        ("fd", FeatureDerivativeTransformer({
            "amount_income_ratio": "amount/income",
            "income_1k": "income/1000",
        })),
        ("cleaner", FeatureCleaner(sentinels=[-999])),
        ("bin_woe", BinnerWoeEncoder(max_bins=6)),
        ("select", IVSelector(iv_threshold=0.02)),
        ("model", model_cls(n_estimators=100, max_depth=4)),
    ]).fit(X, y)


def _bench(fn, n):
    """循环 n 次执行 fn，返回平均单次耗时（us/条）。"""
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = time.perf_counter() - t0
    return dt / n * 1e6


def run_case(name, model_cls, X, y, build_pipe=_build_pipe):
    print()
    print("=" * 62)
    print(f"案例: {name} pipeline")
    print("=" * 62)

    # ---- 1. 训练 + 编译双后端 ----
    pipe = build_pipe(model_cls, X, y)
    print(f"保留特征: {pipe[:-1].transform(X).shape[1]} 个")
    deploy_m2cgen = PipelineParser(backend="m2cgen").compile_pipeline(pipe)
    deploy_onnx = PipelineParser(backend="onnx").compile_pipeline(pipe)
    print(deploy_m2cgen.describe())

    # ---- 2. 生成部署文件（proto+m2cgen / proto+onnx / pkl 参考） ----
    os.makedirs(OUT_DIR, exist_ok=True)
    spec_m2cgen = to_proto_bytes(deploy_m2cgen)
    spec_onnx = to_proto_bytes(deploy_onnx)
    path_m2cgen = os.path.join(OUT_DIR, f"{name}_m2cgen.pbin")
    path_onnx = os.path.join(OUT_DIR, f"{name}_onnx.pbin")
    path_pkl = os.path.join(OUT_DIR, f"{name}_pipe.pkl")
    for p, data in [(path_m2cgen, spec_m2cgen), (path_onnx, spec_onnx)]:
        with open(p, "wb") as f:
            f.write(data)
    with open(path_pkl, "wb") as f:
        pickle.dump(pipe, f)

    # ---- 3. 加载打分器（executor 侧，零 risk_ml 依赖）+ pkl 参考 ----
    scorer_m2cgen = build_engine(spec_m2cgen)
    scorer_onnx = build_engine(spec_onnx)
    with open(path_pkl, "rb") as f:
        pkl_pipe = pickle.load(f)

    # ---- 4. 一致性（以 pkl 为准） ----
    X_check = X.sample(N_CHECK, random_state=0)
    truth = pkl_pipe.predict_proba(X_check)[:, 1]
    consistency = {}
    for label, scorer in [("proto+m2cgen", scorer_m2cgen), ("proto+onnx", scorer_onnx)]:
        pred = scorer.score_rows(X_check.to_dict("records"))
        max_diff = float(np.max(np.abs(pred - truth)))
        n_fail = int(np.sum(np.abs(pred - truth) > ATOL))
        consistency[label] = (max_diff, n_fail)
        status = "PASS" if n_fail == 0 else "FAIL"
        print(f"  一致性[{label:12s}] max_diff={max_diff:.2e} "
              f"fail={n_fail}/{N_CHECK} (atol={ATOL})  [{status}]")

    # ---- 5. 单条打分性能 ----
    row = X_check.to_dict("records")[SINGLE_ROW_IDX]
    row_df = X_check.iloc[[SINGLE_ROW_IDX]]
    # warmup（首次打分触发 onnx session 构建 / pkl 首次 transform 缓存）
    for _ in range(20):
        pkl_pipe.predict_proba(row_df)
        scorer_m2cgen.score(row)
        scorer_onnx.score(row)

    us = {
        "pkl": _bench(lambda: pkl_pipe.predict_proba(row_df), N_PKL_ITERS),
        "proto+m2cgen": _bench(lambda: scorer_m2cgen.score(row), N_ITERS),
        "proto+onnx": _bench(lambda: scorer_onnx.score(row), N_ITERS),
    }
    print("\n  单条打分性能（us/条, QPS, 相对 pkl 加速）:")
    base = us["pkl"]
    for label, t_us in us.items():
        qps = 1e6 / t_us
        speedup = base / t_us
        print(f"    {label:14s} {t_us:10.2f} us | QPS {qps:9.0f} | {speedup:6.1f}x")

    print(f"\n  部署文件（{OUT_DIR}）:")
    for p, desc in [(path_m2cgen, "proto+m2cgen"), (path_onnx, "proto+onnx"), (path_pkl, "pkl")]:
        size = os.path.getsize(p)
        print(f"    {desc:12s} {os.path.basename(p):28s} {size / 1024:8.1f} KB")

    return us, consistency


def main():
    df = make_data(n=5000, seed=42)
    X = df.drop(columns=["y"])
    y = df["y"]
    print(f"数据: {len(df)} 行 x {X.shape[1]} 特征, 正例率 {y.mean():.3f}")

    results = {}
    results["xgb"] = run_case("xgb", RiskXGBClassifier, X, y)
    results["lgb"] = run_case("lgb", RiskLGBMClassifier, X, y)
    results["xgb_fd"] = run_case("xgb_fd", RiskXGBClassifier, X, y,
                                 build_pipe=_build_pipe_fd)

    # ---- 汇总表 ----
    print()
    print("=" * 62)
    print("汇总：单条打分 us/条（一致性均以 pkl 为准）")
    print("=" * 62)
    header = f"{'case':6s} {'backend':12s} {'us/条':>9s} {'QPS':>10s} {'vs pkl':>7s}  一致性"
    print(header)
    for case, (us, consistency) in results.items():
        base = us["pkl"]
        print(f"{case:6s} {'pkl':12s} {us['pkl']:9.2f} {1e6 / us['pkl']:10.0f} {'1.0x':>7s}  —")
        for label in ("proto+m2cgen", "proto+onnx"):
            max_diff, n_fail = consistency[label]
            ok = "PASS" if n_fail == 0 else "FAIL"
            print(f"{case:6s} {label:12s} {us[label]:9.2f} {1e6 / us[label]:10.0f} "
                  f"{base / us[label]:6.1f}x  max_diff {max_diff:.1e} [{ok}]")
    print(f"\n部署文件目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
