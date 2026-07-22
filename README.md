# RiskCraft

风险建模与特征工程框架，包含两大子项目：

- **feature_derivative** — 多端兼容的特征衍生框架（Pandas / PySpark / Dict）
- **risk_ml** — sklearn 兼容的风控建模 ML 框架（清洗 / 分箱 / WOE / 筛选 / 估计器 / 实验对比）

## 快速开始

### 特征衍生

```python
from feature_derivative import transform

# Pandas
import pandas as pd
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
result = transform(df, "a/(a+b)", "ratio")

# Dict（在线服务）
result = transform({"a": 1, "b": 4}, "a/(a+b)", "ratio")
```

### 风控建模

```python
from risk_ml import FeatureCleaner, BinnerWoeEncoder, IVSelector, RiskXGBClassifier

# 典型流水线
cleaner = FeatureCleaner().fit(X_train)
X_clean = cleaner.transform(X_train)

binner_woe = BinnerWoeEncoder().fit(X_clean, y_train)
X_woe = binner_woe.transform(X_clean)

selector = IVSelector().fit(X_woe, y_train)
X_selected = selector.transform(X_woe)

clf = RiskXGBClassifier().fit(X_selected, y_train)
y_score = clf.predict_score(X_selected)
```

### 实验对比

```python
from risk_ml.experiment import (
    ExperimentRunner, ExperimentConfig, TimeWindow,
    make_experiment_grid, AUCMetric, KSMetric, LiftMetric,
)

# 自动生成笛卡尔积配置
configs = make_experiment_grid(
    label_cols=["is_default_30d", "is_default_90d"],
    time_windows=[
        TimeWindow("issue_d", "2024-01-01", "2024-12-31"),
        TimeWindow("issue_d", "2025-01-01", "2025-12-31"),
    ],
)

# 运行实验（含 OOT 跨时间验证 + 多标签评估）
runner = ExperimentRunner(
    configs=configs,
    scoring="ks",
    n_trials=30,
    oot=df_oot,                           # OOT 跨时间验证
    eval_label_cols=["is_default_90d"],    # 多标签评估
)
runner.fit(df_train)

# 查看对比结果
print(runner.results_)

# 最优模型预测
y_score = runner.predict_score(X_test)
```

---

## feature_derivative — 特征衍生框架

接收四则运算表达式字符串，自动解析变量，在三种计算引擎上高效生成新特征列。

### 三端适配

| 引擎 | 输入类型 | 计算方式 | 适用场景 |
|------|---------|---------|---------|
| **Pandas** | `pandas.DataFrame` | `df.eval()` 向量化 | 离线批处理 |
| **PySpark** | `pyspark.sql.DataFrame` | `F.expr()` 分布式 | 大规模数据 |
| **Online** | `dict` | 安全沙箱 `eval()` | 在线推理服务 |

框架根据输入数据类型**自动选择**引擎，无需手动指定。

### 核心特性

- **自动引擎识别** — `transform(data, expression, target_col)` 自动识别数据类型
- **变量校验** — 变量缺失时抛出 `MissingVariableError`，包含缺失字段名
- **安全沙箱**（Online 模式）— AST 节点白名单 + `__builtins__` 置空 + locals 限定
- **缺失值处理** — 传播模式（默认）与预填充模式（`fill_value` 参数）

### 缺失值行为

| 场景 | Pandas | PySpark | Online |
|------|--------|---------|--------|
| 输入含 NaN/None | NaN 传播 | null 传播 | None 传播 |
| 除以零 | inf → NaN | null | None |
| 预填充 `fill_value=0` | `fillna(0)` | `fillna({col: 0})` | None → 0 |

---

## risk_ml — 风控建模框架

sklearn 兼容的风控建模算子工具链，提供特征清洗、分箱、WOE 编码、特征筛选、模型估计、实验对比等全套组件。

### 建模流水线

```
Raw Features → FeatureCleaner → ChiMergeBinner → WoeEncoder
             → IVSelector → CorrelationSelector → PSISelector
             → RiskXGBClassifier
```

### 模块说明

| 模块 | 类 | 说明 |
|------|-----|------|
| **预处理** | `FeatureCleaner` | 哨兵值映射、缺失填充、异常值截断、低质量列删除 |
| **分箱** | `ChiMergeBinner` | 卡方分箱（自底向上合并），支持分类特征 |
| **编码** | `WoeEncoder` / `BinnerWoeEncoder` | WOE 编码，BinnerWoeEncoder 一步到位 |
| **特征筛选** | `IVSelector` / `CorrelationSelector` / `PSISelector` | IV 筛选、相关性去冗余、PSI 稳定性筛选 |
| **估计器** | `RiskXGBClassifier` / `OptunaTuner` | 风控 XGBoost + Optuna 贝叶斯调参 |
| **数据集** | `LendingClubLoader` | Lending Club 贷款数据集自动加载 |
| **实验模块** | `ExperimentRunner` | 多配置实验对比 + OOT 验证 + 多标签评估 |

### 基类体系

所有算子继承自 sklearn 兼容基类：

- **`RiskTransformer`** — 转换器基类（fit / transform），DataFrame-in/DataFrame-out
- **`RiskSelector`** — 筛选器基类（fit / transform + `_get_support_mask`）

### 实验模块

实验组合器支持以下维度的对比：

| 维度 | 参数 | 说明 |
|------|------|------|
| 标签列 | `ExperimentConfig.label_col` | 不同标签定义（30天/90天违约） |
| 时间窗口 | `ExperimentConfig.time_window` | 不同训练时间范围 |
| 样本权重 | `ExperimentConfig.weight_col` | 不同加权策略 |
| OOT 验证 | `ExperimentRunner.oot` | 跨时间样本评估，指标在 OOT 上计算 |
| 多标签评估 | `ExperimentRunner.eval_label_cols` | 同一模型对多个标签分别评估 |
| 自定义指标 | `ExperimentRunner.metrics` | 可扩展的 BaseMetric 体系 |

### 评估指标体系

内置指标，支持自定义扩展：

```python
from risk_ml.experiment import BaseMetric, AUCMetric, KSMetric, LiftMetric

# 自定义指标 — 只需继承 BaseMetric
class GiniMetric(BaseMetric):
    name = "gini"
    def compute(self, y_true, y_score):
        from sklearn.metrics import roc_auc_score
        return 2 * roc_auc_score(y_true, y_score) - 1

runner = ExperimentRunner(
    configs=configs,
    metrics=[AUCMetric(), KSMetric(), LiftMetric(10), GiniMetric()],
)
```

---

## 项目结构

```
RiskCraft/
├── feature_derivative/          # 特征衍生框架
│   ├── __init__.py              # 公共 API & transform() 统一入口
│   ├── exceptions.py            # 自定义异常类
│   ├── parser.py                # 表达式解析（变量提取 & AST 安全校验）
│   ├── sandbox.py               # 安全沙箱 eval()
│   ├── strategies.py            # 策略模式（Pandas / Spark / Online）
│   └── context.py               # 上下文类（自动引擎识别）
├── risk_ml/                     # 风控建模框架
│   ├── _base.py                 # 基类：RiskTransformer / RiskSelector
│   ├── _config.py               # 项目配置（缺失值哨兵值）
│   ├── preprocessing/           # 特征清洗（FeatureCleaner）
│   ├── binning/                 # 分箱（BaseBinner / ChiMergeBinner）
│   ├── encoding/                # WOE 编码（WoeEncoder / BinnerWoeEncoder）
│   ├── feature_selection/       # 特征筛选（IV / Correlation / PSI）
│   ├── estimator/               # 估计器（RiskXGBClassifier / OptunaTuner）
│   ├── dataset/                 # 数据集（LendingClubLoader / demo_data.csv）
│   ├── experiment/              # 实验模块（ExperimentRunner / BaseMetric）
│   │   ├── metrics.py           # 指标基类 & 内置指标（AUC / KS / Lift）
│   │   ├── experiment_config.py # TimeWindow / ExperimentConfig / ExperimentResult
│   │   ├── experiment_runner.py # ExperimentRunner 主类
│   │   └── experiment_grid.py   # make_experiment_grid 笛卡尔积生成
│   └── tests/                   # 测试套件
├── tests/                       # feature_derivative 测试
├── demo.py                      # 特征衍生演示
├── requirements.txt             # 依赖
└── requirements-dev.txt         # 开发依赖
```

## 运行环境

- Python 3.12+
- 核心依赖：numpy / pandas / scikit-learn / xgboost / scipy / optuna

## 运行测试

```bash
# 全量测试
pytest tests/ risk_ml/tests/ -v

# 仅风控建模测试
pytest risk_ml/tests/ -v

# 仅特征衍生测试
pytest tests/ -v
```

## 运行演示

```bash
# 特征衍生演示
python demo.py

# 实验模块演示（使用 demo_data.csv）
python risk_ml/dataset/demo_experiment.py
```
