"""risk_report 模块完整演示 — 使用 demo_data.csv 数据集。

演示流程:
1. 加载 demo_data.csv
2. 按 transaction_time 划分 train(2024前)/test(2024上半年)/oot(2024下半年起)
3. 构建风控建模 Pipeline 并训练
4. 构造 ReportContext
5. 产出完整模型开发文档 Excel
"""

import sys
import time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

print("=" * 60)
print("RiskCraft — risk_report 模块完整演示")
print("=" * 60)

# ============================================================
# 1. 加载数据
# ============================================================
print("\n[1] 加载 demo_data.csv ...")
df = pd.read_csv("risk_ml/dataset/demo_data.csv")
print(f"  总样本: {len(df)}, 欺诈率: {df['is_fraud'].mean():.4f}")

# 解析时间列
df["transaction_time"] = pd.to_datetime(df["transaction_time"])
print(f"  时间范围: {df['transaction_time'].min()} ~ {df['transaction_time'].max()}")

# ============================================================
# 2. 划分数据集
# ============================================================
print("\n[2] 按时间划分 train / test / OOT ...")

# 特征列: 排除 ID/时间/标签 和高基数类别列
exclude_cols = [
    "transaction_id", "customer_id", "transaction_time",
    "is_fraud",  # 标签
    "monthly_spend",  # 高基数, 近似唯一值
]
feature_cols = [c for c in df.columns if c not in exclude_cols]
print(f"  原始特征数: {len(df.columns) - len(exclude_cols)} → {len(feature_cols)}")
print(f"  特征列表: {feature_cols}")

# 时间划分
# train: 2023年10月 ~ 2023年12月 (早期数据,建模用)
# test:  2024年1月 ~ 2024年6月 (近期数据,测试用)
# oot:   2024年7月 ~ 2026年6月 (跨时间验证)
train_mask = df["transaction_time"] < "2024-01-01"
test_mask = (df["transaction_time"] >= "2024-01-01") & (df["transaction_time"] < "2024-07-01")
oot_mask = df["transaction_time"] >= "2024-07-01"

X_train = df.loc[train_mask, feature_cols].copy()
y_train = df.loc[train_mask, "is_fraud"].values
X_test = df.loc[test_mask, feature_cols].copy()
y_test = df.loc[test_mask, "is_fraud"].values
X_oot = df.loc[oot_mask, feature_cols].copy()
y_oot = df.loc[oot_mask, "is_fraud"].values

print(f"  train: {len(X_train)} 条, 欺诈率 {y_train.mean():.4f}")
print(f"  test:  {len(X_test)} 条, 欺诈率 {y_test.mean():.4f}")
print(f"  oot:   {len(X_oot)} 条, 欺诈率 {y_oot.mean():.4f}")

# ============================================================
# 3. 构建 Pipeline 并训练
# ============================================================
print("\n[3] 构建风控建模 Pipeline 并训练 ...")

from sklearn.pipeline import Pipeline
from risk_ml.preprocessing import FeatureCleaner
from risk_ml.encoding import BinnerWoeEncoder
from risk_ml.feature_selection import IVSelector, CorrelationSelector
from risk_ml.estimator import RiskXGBClassifier, OptunaTuner

# 使用 OptunaTuner(RiskXGBClassifier()) 作为分类器步骤
# OptunaTuner 继承 BaseEstimator，完全兼容 sklearn Pipeline
# fit 时自动调参: n_trials=20 次贝叶斯搜索，评估指标用 KS
pipe = Pipeline([
    ("cleaner", FeatureCleaner()),
    ("binner_woe", BinnerWoeEncoder(max_bins=8)),
    ("iv_selector", IVSelector(iv_threshold=0.02)),
    ("corr_selector", CorrelationSelector(corr_threshold=0.7)),
    ("classifier", OptunaTuner(
        estimator=RiskXGBClassifier(),
        n_trials=20,
        scoring="ks",
        cv=3,
        random_state=42,
        verbose=0,
    )),
])

start = time.time()
pipe.fit(X_train, y_train)
elapsed = time.time() - start
print(f"  训练完成, 耗时 {elapsed:.1f}s")

# OptunaTuner fit 后，最优参数和最优估计器
tuner = pipe.named_steps["classifier"]
print(f"  Optuna 最优参数: {tuner.best_params_}")
print(f"  Optuna 最优 KS: {tuner.best_score_:.4f}")

# 查看入模特征 — 从 best_estimator_ 获取
best_est = tuner.best_estimator_
final_features = best_est.feature_names_in_
print(f"  入模特征数: {len(final_features)}")
print(f"  入模特征: {list(final_features)}")

# 查看关键属性
bwe = pipe.named_steps["binner_woe"]
iv_sel = pipe.named_steps["iv_selector"]
print(f"\n  IV 值 (WOE编码后):")
if hasattr(iv_sel, "iv_values_"):
    for feat, iv in iv_sel.iv_values_.items():
        print(f"    {feat}: IV={iv:.4f}")

# ============================================================
# 4. 构造 ReportContext
# ============================================================
print("\n[4] 构造 ReportContext ...")

from risk_report import ReportContext

# 特征元信息
feature_meta = {
    "transaction_amount": {"含义": "交易金额", "来源": "交易系统", "类别": "数值"},
    "merchant_category": {"含义": "商户类别", "来源": "商户信息", "类别": "分类"},
    "transaction_type": {"含义": "交易类型", "来源": "交易系统", "类别": "分类"},
    "payment_method": {"含义": "支付方式", "来源": "交易系统", "类别": "分类"},
    "city": {"含义": "城市", "来源": "用户信息", "类别": "分类"},
    "country": {"含义": "国家", "来源": "用户信息", "类别": "分类"},
    "device_type": {"含义": "设备类型", "来源": "交易日志", "类别": "分类"},
    "operating_system": {"含义": "操作系统", "来源": "交易日志", "类别": "分类"},
    "browser": {"含义": "浏览器", "来源": "交易日志", "类别": "分类"},
    "card_type": {"含义": "卡类型", "来源": "卡片信息", "类别": "分类"},
    "card_present": {"含义": "是否现场刷卡", "来源": "交易系统", "类别": "二元"},
    "international_transaction": {"含义": "是否跨境交易", "来源": "交易系统", "类别": "二元"},
    "distance_from_home": {"含义": "离家距离", "来源": "位置数据", "类别": "数值"},
    "previous_transaction_gap": {"含义": "上次交易间隔", "来源": "交易日志", "类别": "数值"},
    "daily_transaction_count": {"含义": "当日交易次数", "来源": "交易日志", "类别": "数值"},
    "risk_score": {"含义": "风控评分", "来源": "风控引擎", "类别": "数值"},
    "customer_age": {"含义": "客户年龄", "来源": "客户信息", "类别": "数值"},
    "account_tenure_years": {"含义": "账户年限", "来源": "客户信息", "类别": "数值"},
    "merchant_risk_level": {"含义": "商户风险等级", "来源": "风控引擎", "类别": "分类"},
}

context = ReportContext(
    model_name="反欺诈模型_v1.0",
    developer="RiskCraft Demo",
    validator="风控团队",
    business_owner="业务运营部",
    background="针对线上交易欺诈风险，构建反欺诈预测模型，降低欺诈损失率",
    application="线上交易实时风控筛查，辅助人工审核决策",
    pipeline=pipe,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    X_oot=X_oot,
    y_oot=y_oot,
    label_definition={0: "正常", 1: "欺诈"},
    feature_meta=feature_meta,
)

print(f"  pipeline_attrs.feature_names_in_: {context.pipeline_attrs.feature_names_in_}")
print(f"  y_score_train shape: {context.y_score_train.shape}")
print(f"  y_score_test shape: {context.y_score_test.shape if context.y_score_test is not None else 'None'}")
print(f"  y_score_oot shape: {context.y_score_oot.shape if context.y_score_oot is not None else 'None'}")

# ============================================================
# 5. 产出完整模型开发文档
# ============================================================
print("\n[5] 产出完整模型开发文档 Excel ...")

from risk_report import ModelReport

report = ModelReport()
report.fit(context)

print(f"  生成 sections: {list(report.results_.keys())}")

output_path = "risk_report/demo_report_output.xlsx"
report.to_excel(output_path)
print(f"  ✓ Excel 已保存: {output_path}")

# 验证 Excel 内容
import openpyxl
wb = openpyxl.load_workbook(output_path)
print(f"\n  Excel Sheet 概览:")
print(f"  {'Sheet名':<20s} {'行数':>6s} {'列数':>6s}")
print(f"  {'-'*20} {'-'*6} {'-'*6}")
for name in wb.sheetnames:
    ws = wb[name]
    print(f"  {name:<20s} {ws.max_row:>6d} {ws.max_column:>6d}")

# ============================================================
# 6. 快速查看各数据集指标
# ============================================================
print("\n[6] 各数据集模型效果:")
from risk_report import ModelEffectOperator

datasets = {}
if context.y_train is not None and context.y_score_train is not None:
    datasets["训练集"] = (context.y_train, context.y_score_train)
if context.y_test is not None and context.y_score_test is not None:
    datasets["测试集"] = (context.y_test, context.y_score_test)
if context.y_oot is not None and context.y_score_oot is not None:
    datasets["OOT验证集"] = (context.y_oot, context.y_score_oot)

effect_df = ModelEffectOperator.compute_effect_table(datasets)
print(effect_df.to_string(index=False))

print("\n" + "=" * 60)
print("演示完成! 报告文件: risk_report/demo_report_output.xlsx")
print("=" * 60)
