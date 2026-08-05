# RiskCraft

风险建模与特征工程框架，包含三大子项目：

- **feature_derivative** — 多端兼容的特征衍生框架（Pandas / PySpark / Dict）
- **risk_ml** — sklearn 兼容的风控建模 ML 框架（清洗 / 分箱 / WOE / 筛选 / 估计器 / 实验对比）
- **risk_report** — 模型报告自动产出模块（22 算子 / 8 Sheet / 配置驱动）

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
             → RiskXGBClassifier / RiskLGBMClassifier
```

### 模块说明

| 模块 | 类 | 说明 |
|------|-----|------|
| **预处理** | `FeatureCleaner` | 哨兵值映射、缺失填充、异常值截断、低质量列删除 |
| **分箱** | `ChiMergeBinner` | 卡方分箱（自底向上合并），支持分类特征 |
| **编码** | `WoeEncoder` / `BinnerWoeEncoder` | WOE 编码，BinnerWoeEncoder 一步到位 |
| **特征筛选** | `IVSelector` / `CorrelationSelector` / `PSISelector` | IV 筛选、相关性去冗余、PSI 稳定性筛选 |
| **估计器** | `RiskXGBClassifier` / `RiskLGBMClassifier` / `OptunaTuner` | 风控 XGB / LGB + Optuna 贝叶斯调参 |
| **数据集** | `LendingClubLoader` | Lending Club 贷款数据集自动加载 |
| **实验模块** | `ExperimentRunner` | 多配置实验对比 + OOT 验证 + 多标签评估 |

### 基类体系

所有算子继承自 sklearn 兼容基类。四类部署模块的基类即**在线扩展契约**，
新增子类（自定义分箱 / 编码 / 筛选 / 估计器）后在线部署零改动即可编译上线：

| 模块 | 基类（子类需实现） | 部署产物 |
|------|--------------------|----------|
| 分箱 | `BaseBinner`（`_bin_column`） | `BinOp` |
| 编码 | `BaseEncoder`（`fit` 产出 `woe_map_`） | `WoeOp` / `BinWoeOp` |
| 筛选 | `RiskSelector`（`_get_support_mask`） | `SelectOp` |
| 估计器 | `RiskEstimator`（`to_deploy_model` → TreeModel） | 树模型后端（m2cgen / onnx，xgb / lgb） |

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

## risk_report — 模型报告自动产出模块

从已拟合的 Pipeline + 单 DataFrame 自动产出标准模型开发报告（8 Sheet / 22 算子），配置驱动、可扩展。

### 三种使用模式

```python
from risk_report import (
    ModelReport, ReportContext, ExcelWriter,
    ScoreLiftOperator, compute_lift_table,
    DEFAULT_DOCUMENT_CONFIG, DocumentConfig, SheetConfig,
)

# 1. 日常单独调用（无需构造 ReportContext）
df = ScoreLiftOperator.compute_lift_table(y_true, y_score, n_bins=10)

# 2. 模块化组装（自定义配置）
config = DocumentConfig(sheets=[
    SheetConfig("模型表现", [ScoreLiftOperator(), ModelEffectOperator()]),
])
report = ModelReport(config=config).fit(context).to_excel("report.xlsx")

# 3. 全量报告（默认 8 Sheet / 22 算子）
context = ReportContext(
    data=df, tag_col="tag", label_col="is_fraud",
    pipeline=fitted_pipeline, time_col="transaction_time",
)
ModelReport().fit(context).to_excel("report.xlsx")
```

### 数据输入

单 DataFrame + tag_col + label_col，替代传统 X_train/y_train/X_test/... 六数组：

```python
context = ReportContext(
    data=df,                  # 包含 tag/label/score 的完整 DataFrame
    tag_col="tag",            # 区分 train/test/oot
    label_col="is_fraud",     # 主标签列
    pipeline=pipe,            # 已拟合流水线（自动提取属性和预测分数）
    score_col="score",        # 模型分数列（可由 pipeline 自动计算）
    baseline_score_col="baseline_score",  # 对标模型分数列
    time_col="transaction_time",          # 时间列（月度拆分分析）
    extra_labels=["y_mob3", "y_mob6"],    # 多标签列（MOB 压测）
)
```

### 22 算子一览

| Sheet | 算子 | 说明 |
|-------|------|------|
| 模型说明 | MetaInfo | 模型元信息 |
| 1.模型设计 | DevPurpose / ModelAssumption / LabelDefinition / SampleSelection / ModelingSample / EffectSummary | 模型设计文档 |
| 2.变量分析 | VarDescription / VarCleaning / VarFilter / VarAnalysis | 变量描述、清洗、筛选、IV/KS/Gain分析 |
| 附件-变量分箱 | VarBinning | 逐变量分箱 WOE 明细表 |
| 3.模型表现 | ModelMethod / ModelEffect / ScoreLift / ScoreLiftGray | 模型方法、指标表、分箱表现、含灰分箱 |
| 附件1-补充分析 | Attribution / ModelComparison / MobPerformance / Portrait | 归因、模型对比、MOB压测、画像 |
| 附件2-模型使用方案 | SwapAnalysis | 切分点分析 |
| 附件3-变量描述 | VarRange | 变量取值范围 |

### Excel 美化

- 微软雅黑字体 + 交替行色
- 百分比列自动格式化（`xx.xx%`）
- lift/bad_rate 等列条件格式数据条
- 表头深蓝背景 + 白色加粗
- 仅变量分析和变量描述 Sheet 冻结表头

### IV 计算性能优化

IV 计算核心函数 `compute_iv_from_data()` 采用 numba JIT + 并行加速：

| 方案 | 100列×50K行 | 200列×100K行 |
|---|---|---|
| numba parallel | 0.066s | 0.299s |
| numpy bincount（fallback） | 0.306s | 1.227s |
| 原始 Python | >60s | >120s |

numba 不可用时自动降级到 bincount 方案，跨平台兼容（Windows/Linux/macOS）。

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
│   ├── _pipeline.py             # RiskPipeline（sklearn Pipeline 子类）
│   ├── _config.py               # 项目配置（缺失值哨兵值）
│   ├── preprocessing/           # 特征清洗（FeatureCleaner）
│   ├── binning/                 # 分箱（BaseBinner / ChiMergeBinner）
│   ├── encoding/                # WOE 编码（WoeEncoder / BinnerWoeEncoder）
│   ├── feature_selection/       # 特征筛选（IV / Correlation / PSI）
│   ├── estimator/               # 估计器（RiskXGBClassifier / RiskLGBMClassifier / OptunaTuner）
│   ├── dataset/                 # 数据集（LendingClubLoader / demo_data.csv）
│   ├── experiment/              # 实验模块（ExperimentRunner / BaseMetric）
│   │   ├── metrics.py           # 指标基类 & 内置指标（AUC / KS / Lift）
│   │   ├── experiment_config.py # TimeWindow / ExperimentConfig / ExperimentResult
│   │   ├── experiment_runner.py # ExperimentRunner 主类
│   │   └── experiment_grid.py   # make_experiment_grid 笛卡尔积生成
│   └── tests/                   # 测试套件
├── risk_report/                 # 模型报告自动产出模块
│   ├── __init__.py              # 公共 API（22 算子 + 配置类 + 上下文类）
│   ├── _base.py                 # 基类：ReportOperator / SubSection / placeholder_df
│   ├── _context.py              # 报告上下文：ReportContext / PipelineAttributes
│   ├── _excel.py                # Excel 写入器（美化 + 数据条 + 交替行色）
│   ├── _format.py               # 格式配置：FormatConfig / DEFAULT_FORMAT
│   ├── _scoring.py              # 计算工具（lift/swap/ks/stats/iv + numba加速）
│   ├── _config.py               # SheetConfig / DocumentConfig
│   ├── _templates.py            # DEFAULT_DOCUMENT_CONFIG（8 Sheet / 22 算子）
│   ├── report.py                # 组合器：ModelReport
│   ├── operators/               # 22 算子模块
│   ├── tests/                   # 测试套件
│   └── demo_full_pipeline.py    # 全流程演示脚本
├── tests/                       # feature_derivative 测试
├── demo.py                      # 特征衍生演示
├── requirements.txt             # 依赖
└── requirements-dev.txt         # 开发依赖
```

## 运行环境

- Python 3.12+
- 核心依赖：numpy / pandas / scikit-learn / xgboost / scipy / optuna / openpyxl
- 可选加速：numba（IV 计算并行加速，200 列 × 10 万行 < 0.5s）

## 运行测试

```bash
# 全量测试
pytest tests/ risk_ml/tests/ risk_report/tests/ -v

# 仅风控建模测试
pytest risk_ml/tests/ -v

# 仅特征衍生测试
pytest tests/ -v

# 仅报告模块测试
pytest risk_report/tests/ -v
```

## 运行演示

```bash
# 特征衍生演示
python demo.py

# 实验模块演示（使用 demo_data.csv）
python risk_ml/dataset/demo_experiment.py

# 全流程报告演示（数据划分 → Pipeline训练 → 报告产出）
python risk_report/demo_full_pipeline.py
```

---

## 更新日志

### v0.3.0 — risk_report 报告模块 + IV 性能优化

- **新增 risk_report 独立顶层包**：22 算子 / 8 Sheet 配置驱动报告，从 risk_ml.report 迁移为独立包
- **单 DataFrame 输入**：`ReportContext(data, tag_col, label_col)` 替代传统六数组
- **全流程 Demo**：`demo_full_pipeline.py` — 数据划分 → Pipeline 训练 → 报告产出
- **月度拆分分析**：model_effect / score_lift 算子支持全量数据集按月拆分
- **IV 统一算法**：`compute_iv_from_data()` 单入口，BinnerWoeEncoder / IVSelector / 自动计算数值一致
- **IV 自动兜底**：Pipeline 无 WOE/IV 步骤时自动从 data 计算 IV
- **IV 性能优化**：numba JIT + 并行加速，200 列 × 10 万行 < 0.3s（原 >120s）
- **Excel 美化**：微软雅黑字体、交替行色、百分比格式化、数据条、选择性冻结表头
- **样本选择自动计算**：开发样本分布自动从 context.data 计算，原始样本分布提供简化版
- **移除 baseline 对比**：model_effect / score_lift 算子移除 baseline 对标列

### v0.2.0 — risk_ml 实验模块

- 新增 ExperimentRunner：多配置实验对比 + OOT 验证 + 多标签评估
- 新增 TimeWindow / ExperimentConfig / ExperimentResult 配置体系
- 新增 BaseMetric 指标体系：AUC / KS / Lift 可扩展
- 新增 make_experiment_grid 笛卡尔积配置生成
- 新增 RiskPipeline（sklearn Pipeline 子类）

### v0.1.0 — 初始版本

- feature_derivative 多端兼容特征衍生框架（Pandas / PySpark / Dict）
- risk_ml sklearn 兼容风控建模框架（清洗 / 分箱 / WOE / 筛选 / 估计器）
- ChiMergeBinner 卡方分箱
- BinnerWoeEncoder 分箱 + WOE 联合算子
- OptunaTuner 贝叶斯调参
