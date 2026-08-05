"""
实验模块演示 — 使用 demo_data.csv 交易数据

实验设计：
1. 训练集 vs OOT（最近3个月跨时间验证）
2. 不同时间窗口：2024年 vs 2025年
3. 不同样本权重：等权 vs 高风险商户加权
4. 多标签评估：is_fraud + is_high_risk
5. OOT 指标驱动 best 选举

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

from risk_ml import FeatureCleaner, RiskXGBClassifier, RiskPipeline
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
# 2. 加载与切分数据
# ============================================================
print("=" * 70)
print("[1] 加载与切分数据")
print("=" * 70)

df = pd.read_csv("risk_ml/dataset/demo_data.csv")
df["transaction_time"] = pd.to_datetime(df["transaction_time"])

print(f"原始数据: {df.shape[0]:,} 行, {df.shape[1]} 列")
print(f"欺诈率:   {df['is_fraud'].mean():.4f} ({df['is_fraud'].sum():,} 笔欺诈)")
print(f"时间范围: {df['transaction_time'].min().date()} ~ {df['transaction_time'].max().date()}")

# 切分：最近3个月为 OOT，其余为训练集
oot_cutoff = df["transaction_time"].max() - pd.DateOffset(months=3)
df_train = df[df["transaction_time"] < oot_cutoff].copy()
df_oot = df[df["transaction_time"] >= oot_cutoff].copy()

print(f"\nOOT 切分点: {oot_cutoff.date()}")
print(f"训练集: {len(df_train):,} 行, 欺诈率 {df_train['is_fraud'].mean():.4f}")
print(f"OOT集:  {len(df_oot):,} 行, 欺诈率 {df_oot['is_fraud'].mean():.4f}")

# 采样 10% 加速演示
df_train = df_train.sample(frac=0.1, random_state=42).reset_index(drop=True)
df_oot = df_oot.sample(frac=0.1, random_state=42).reset_index(drop=True)
print(f"\n采样后 — 训练: {len(df_train):,} 行, OOT: {len(df_oot):,} 行")

# ============================================================
# 3. 特征工程
# ============================================================
print("\n" + "=" * 70)
print("[2] 特征工程")
print("=" * 70)

# 构造样本权重：高风险商户权重更高
weight_map = {"High": 3.0, "Medium": 1.5, "Low": 1.0}
df_train["risk_weight"] = df_train["merchant_risk_level"].map(weight_map).fillna(1.0)
df_oot["risk_weight"] = df_oot["merchant_risk_level"].map(weight_map).fillna(1.0)

# 构造第二个标签：高风险交易（risk_score >= 70 且欺诈）
df_train["is_high_risk"] = ((df_train["risk_score"] >= 70) & (df_train["is_fraud"] == 1)).astype(int)
df_oot["is_high_risk"] = ((df_oot["risk_score"] >= 70) & (df_oot["is_fraud"] == 1)).astype(int)

print(f"is_fraud 训练: {dict(df_train['is_fraud'].value_counts())}")
print(f"is_fraud OOT:  {dict(df_oot['is_fraud'].value_counts())}")
print(f"is_high_risk 训练: {dict(df_train['is_high_risk'].value_counts())}")
print(f"is_high_risk OOT:  {dict(df_oot['is_high_risk'].value_counts())}")

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

# ============================================================
# 4. 实验配置
# ============================================================
print("\n" + "=" * 70)
print("[3] 实验配置")
print("=" * 70)

tw_2024 = TimeWindow("transaction_time", "2024-01-01", "2024-12-31")
tw_2025 = TimeWindow("transaction_time", "2024-01-01", "2025-12-31")

configs = [
    # 基线：全量训练数据，等权
    ExperimentConfig(name="baseline", label_col="is_fraud"),
    # 2024 年窗口
    ExperimentConfig(name="2024", label_col="is_fraud", time_window=tw_2024),
    # 2025 年窗口
    ExperimentConfig(name="2025", label_col="is_fraud", time_window=tw_2025),
    # 高风险商户加权
    ExperimentConfig(
        name="2025_weighted", label_col="is_fraud",
        time_window=tw_2025, weight_col="risk_weight",
    ),
]

for c in configs:
    parts = [f"  {c.name}: label={c.label_col}"]
    if c.time_window:
        parts.append(f"window={c.time_window}")
    if c.weight_col:
        parts.append(f"weight={c.weight_col}")
    print(", ".join(parts))

# ============================================================
# 5. 运行实验（OOT holdout 调参 + 多标签评估）
# ============================================================
print("\n" + "=" * 70)
print("[4] 运行实验（RiskPipeline + OOT holdout 调参 + 多标签评估）")
print("=" * 70)

pipe = RiskPipeline([
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
    n_trials=10,
    tuner_cv=3,
    oot=df_oot,                             # OOT 跨时间验证
    eval_label_cols=["is_high_risk"],        # 多标签评估
    n_jobs=1,
    random_state=42,
    verbose=1,
)

runner.fit(df_train)
print(runner.show())

# # ============================================================
# # 6. 展示结果（Markdown 报告）
# # ============================================================
# print("\n" + "=" * 70)
# print("[5] 实验结果 Markdown 报告")
# print("=" * 70)

# report = runner.show(top_n_features=10)
# print(report)

# # 最优模型在 OOT 上的预测示例
# print("\n最优模型 OOT 前 10 笔预测示例:")
# X_oot_sample = df_oot[feature_cols].head(10)
# scores = runner.predict_score(X_oot_sample)
# for i, s in enumerate(scores):
#     label = "欺诈" if df_oot["is_fraud"].iloc[i] == 1 else "正常"
#     print(f"  样本{i+1}: 预测概率={s:.4f}, 实际={label}")

# print("\n" + "=" * 70)
# print("[完成] 实验模块演示结束")
# print("=" * 70)
