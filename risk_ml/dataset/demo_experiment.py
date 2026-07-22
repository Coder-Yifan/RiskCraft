"""
实验模块演示 — 使用 demo_data.csv 交易数据

实验设计：
1. 不同时间窗口：2024年 vs 2025年 vs 全量
2. 不同样本权重：等权 vs 高风险商户加权
3. 对比 Optuna 调参后的模型表现

运行方式：
    python risk_ml/dataset/demo_experiment.py
"""

import sys
import io
import os

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Windows GBK 终端兼容
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from risk_ml import FeatureCleaner, RiskXGBClassifier
from risk_ml.experiment import (
    ExperimentRunner,
    ExperimentConfig,
    TimeWindow,
    make_experiment_grid,
    AUCMetric,
    KSMetric,
    LiftMetric,
    BaseMetric,
)


# ============================================================
# 1. 自定义指标：Gini 系数
# ============================================================
class GiniMetric(BaseMetric):
    """Gini 系数 = 2 * AUC - 1，风控评分卡常用指标"""

    name = "gini"

    def compute(self, y_true, y_score):
        from sklearn.metrics import roc_auc_score

        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        if len(np.unique(y_true)) < 2:
            return 0.0
        return 2 * roc_auc_score(y_true, y_score) - 1


# ============================================================
# 2. 加载数据
# ============================================================
print("=" * 70)
print("[1] 加载数据")
print("=" * 70)

df = pd.read_csv("risk_ml/dataset/demo_data.csv")
print(f"原始数据: {df.shape[0]} 行, {df.shape[1]} 列")
print(f"欺诈率: {df['is_fraud'].mean():.4f} ({df['is_fraud'].sum()} 笔欺诈)")
print(f"时间范围: {df['transaction_time'].min()} ~ {df['transaction_time'].max()}")

# 采样 10% 加速演示（全量数据跑完需要较长时间）
df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)
print(f"采样后: {df.shape[0]} 行, 欺诈率: {df['is_fraud'].mean():.4f}")

# ============================================================
# 3. 特征工程
# ============================================================
print("\n" + "=" * 70)
print("[2] 特征工程")
print("=" * 70)

# 解析时间列
df["transaction_time"] = pd.to_datetime(df["transaction_time"])

# 构造样本权重：高风险商户权重更高
weight_map = {"High": 3.0, "Medium": 1.5, "Low": 1.0}
df["risk_weight"] = df["merchant_risk_level"].map(weight_map).fillna(1.0)

# 构造第二个标签：高风险交易（risk_score >= 70 且欺诈）
df["is_high_risk"] = ((df["risk_score"] >= 70) & (df["is_fraud"] == 1)).astype(int)

print(f"is_fraud 分布: {dict(df['is_fraud'].value_counts())}")
print(f"is_high_risk 分布: {dict(df['is_high_risk'].value_counts())}")
print(f"risk_weight 分布: {dict(df['risk_weight'].value_counts())}")

# 选取数值型特征列
feature_cols = [
    "transaction_amount",
    "card_present",
    "international_transaction",
    "distance_from_home",
    "previous_transaction_gap",
    "daily_transaction_count",
    "monthly_spend",
    "risk_score",
    "customer_age",
    "account_tenure_years",
]
print(f"特征数: {len(feature_cols)}")
print(f"特征列: {feature_cols}")

# ============================================================
# 4. 定义实验配置
# ============================================================
print("\n" + "=" * 70)
print("[3] 实验配置")
print("=" * 70)

# 时间窗口
tw_2024 = TimeWindow("transaction_time", "2024-01-01", "2024-12-31")
tw_2025 = TimeWindow("transaction_time", "2025-01-01", "2025-12-31")

# 手动定义实验配置
configs = [
    # 基线实验：全量数据，等权
    ExperimentConfig(name="baseline_full", label_col="is_fraud"),
    # 2024 年窗口
    ExperimentConfig(name="2024_equal", label_col="is_fraud", time_window=tw_2024),
    # 2025 年窗口
    ExperimentConfig(name="2025_equal", label_col="is_fraud", time_window=tw_2025),
    # 高风险商户加权
    ExperimentConfig(
        name="2024_weighted", label_col="is_fraud",
        time_window=tw_2024, weight_col="risk_weight",
    ),
    # 不同标签定义：高风险交易
    ExperimentConfig(
        name="2025_high_risk", label_col="is_high_risk", time_window=tw_2025,
    ),
]

for c in configs:
    parts = [f"  {c.name}: label={c.label_col}"]
    if c.time_window:
        parts.append(f"time={c.time_window}")
    if c.weight_col:
        parts.append(f"weight={c.weight_col}")
    print(", ".join(parts))

# ============================================================
# 5. 构建 Pipeline & 运行实验
# ============================================================
print("\n" + "=" * 70)
print("[4] 运行实验（Optuna 自动调参）")
print("=" * 70)

pipe = Pipeline([
    ("cleaner", FeatureCleaner()),
    ("classifier", RiskXGBClassifier(n_estimators=50)),
])

metrics = [AUCMetric(), KSMetric(), LiftMetric(percentile=10), GiniMetric()]

runner = ExperimentRunner(
    configs=configs,
    pipeline=pipe,
    feature_columns=feature_cols,
    metrics=metrics,
    scoring="ks",
    n_trials=10,      # 演示用 10 轮，实际建议 30-50
    tuner_cv=3,
    n_jobs=1,
    random_state=42,
    verbose=1,
)

runner.fit(df)

# ============================================================
# 6. 展示结果
# ============================================================
print("\n" + "=" * 70)
print("[5] 实验对比结果")
print("=" * 70)

# 格式化输出
results = runner.results_[
    ["name", "label_col", "time_window", "weight_col", "status",
     "n_samples", "default_rate", "n_features",
     "auc", "ks", "lift_10", "gini",
     "best_trial_score", "training_time"]
].copy()

# 格式化浮点数
for col in ["default_rate", "auc", "ks", "lift_10", "gini", "best_trial_score", "training_time"]:
    results[col] = results[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
results["n_samples"] = results["n_samples"].apply(lambda x: f"{x:,}")

print(results.to_string(index=False))

# ============================================================
# 7. 最优实验
# ============================================================
print("\n" + "=" * 70)
print("[6] 最优实验")
print("=" * 70)

print(f"实验名: {runner.best_config_.name}")
print(f"标签列: {runner.best_config_.label_col}")
print(f"KS 值:  {runner.best_score_:.4f}")

# 最优模型的前 10 笔预测
print("\n最优模型前 10 笔预测示例:")
X_sample = df[feature_cols].head(10)
scores = runner.predict_score(X_sample)
for i, s in enumerate(scores):
    label = "欺诈" if df["is_fraud"].iloc[i] == 1 else "正常"
    print(f"  样本{i+1}: 预测概率={s:.4f}, 实际={label}")

print("\n" + "=" * 70)
print("[完成] 实验模块演示结束")
print("=" * 70)
