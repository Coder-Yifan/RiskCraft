# RiskCraft API 参考文档

> 风险建模与特征工程框架 — 完整 API 参考
>
> 版本：1.0.0 ｜ 环境：Python 3.12 ｜ 更新日期：2026-08-03

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 安装与环境](#2-安装与环境)
- [3. feature_derivative — 特征衍生框架](#3-feature_derivative--特征衍生框架)
- [4. risk_ml — 风控建模框架](#4-risk_ml--风控建模框架)
- [5. risk_report — 模型报告模块](#5-risk_report--模型报告模块)
- [6. 端到端实战示例](#6-端到端实战示例)
- [7. 附录](#7-附录)
- [8. 进阶开发指南](#8-进阶开发指南)

---

## 1. 项目概述

RiskCraft 是一个面向风控场景的建模与特征工程框架，由三个独立顶层包组成：

| 包 | 定位 | 规模 | 关键能力 |
|---|---|---|---|
| **feature_derivative** | 特征衍生 | ~700 行 | 四则运算表达式，Pandas / PySpark / Online 三端自动适配 |
| **risk_ml** | 风控建模 | ~3400 行 | 清洗 / 分箱 / WOE / 筛选 / 估计器 / 实验对比，sklearn 兼容 |
| **risk_report** | 模型报告 | ~4100 行 | 配置驱动，22 算子 / 8 Sheet 自动产出标准模型开发报告 |

三者关系：`risk_ml` 的 IV / 相关性计算委托给 `risk_report._scoring`（统一算法），`risk_ml` 根包通过 `__getattr__` 懒加载重导出 `risk_report` 符号以保持向后兼容。

---

## 2. 安装与环境

### 2.1 运行环境

| 项 | 要求 |
|---|---|
| Python | 3.12.4 |
| numpy / pandas | 1.26.4 / 2.2.2 |
| scikit-learn | 1.5.2 |
| xgboost | 2.1.3 |
| optuna | 3.6.1 |
| numba | ≥0.60（可选，IV 计算加速） |
| openpyxl | 3.1.5（报告 Excel 输出） |

### 2.2 安装依赖

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

### 2.3 可选依赖

- **PySpark**：仅 `feature_derivative` 的 Spark 引擎需要（框架未安装时自动跳过，不影响其他功能）
- **numba**：IV 计算 JIT 加速，未安装时自动回退到 bincount 实现
- **kaggle**：`LendingClubLoader` 自动下载数据集需要

---

## 3. feature_derivative — 特征衍生框架

### 3.1 快速上手

```python
import pandas as pd
from feature_derivative import transform

# Pandas（离线批处理）
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
df = transform(df, "a/(a+b)", "ratio")       # 返回新 DataFrame（含 ratio 列）

# dict（在线服务单条）
result = transform({"a": 1, "b": 4}, "a/(a+b)", "ratio")

# 预填充缺失值
result = transform(df, "a/(a+b)", "ratio", fill_value=0)
```

框架根据输入数据类型**自动选择引擎**，无需手动指定。

### 3.2 统一入口 `transform`

```python
transform(data, expression: str, target_col: str, fill_value=None)
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `data` | `pd.DataFrame` / `pyspark.sql.DataFrame` / `dict` | 必填 | 输入数据，自动识别类型选择引擎 |
| `expression` | `str` | 必填 | 四则运算表达式，如 `"a/(a+b)"`，支持 `+ - * /` 与括号 |
| `target_col` | `str` | 必填 | 新特征列名 |
| `fill_value` | `float` | `None` | 缺失值填充值；`None` 传播（NaN/null/None 原样传播），数值则计算前预填充 |

**返回值**：类型与输入一致（DataFrame → DataFrame，dict → dict）。

**抛出的异常**：

| 异常 | 触发条件 |
|---|---|
| `MissingVariableError` | 表达式变量在数据中缺失 |
| `ExpressionSyntaxError` | 表达式语法错误 |
| `UnsafeExpressionError` | 表达式不安全（**仅 Online 模式**触发） |
| `TypeError` | 不支持的数据类型 |

### 3.3 三端适配差异

| 维度 | PandasStrategy | SparkStrategy | OnlineStrategy |
|---|---|---|---|
| 输入类型 | `pandas.DataFrame` | `pyspark.sql.DataFrame` | `dict` |
| 计算方式 | `df.eval()` 向量化 | `F.expr()` 分布式 SQL | 安全沙箱 `eval()` 单条 |
| 缺失值 | NaN 传播 | null 传播 | None 传播 |
| 除以零 | `inf`/`-inf` → NaN | → null | → None |
| 原数据是否修改 | 否（操作副本） | 否（返回新 DF） | 否（返回新 dict） |
| 可触发不安全异常 | 否 | 否 | **是** |

### 3.4 公开 API 清单

**包导出**（`from feature_derivative import ...`，11 个符号）：

| 类别 | 符号 |
|---|---|
| 入口 | `transform` |
| 上下文 | `FeatureDerivativeContext` |
| 策略类 | `PandasStrategy` / `SparkStrategy` / `OnlineStrategy` |
| 异常 | `FeatureDerivativeError` / `MissingVariableError` / `ExpressionSyntaxError` / `UnsafeExpressionError` |
| 工具 | `extract_variables` / `validate_ast_safety` |

#### `FeatureDerivativeContext`

策略模式上下文，根据数据类型自动选择策略。`transform(data, expression, target_col, fill_value=None)` 是其薄封装。

#### `extract_variables(expression: str) -> List[str]`

提取表达式中的变量名（去重、按首次出现顺序）。抛出 `ExpressionSyntaxError`。

```python
extract_variables("((x+y)*z)/(x-y)")   # → ['x', 'y', 'z']
```

#### `validate_ast_safety(expression: str) -> ast.AST`

校验 AST 节点白名单。返回解析后的 AST 树；语法错误抛 `ExpressionSyntaxError`，含非白名单节点抛 `UnsafeExpressionError`。

#### `safe_eval_expression(expression: str, variables: dict)`

安全沙箱单条计算（**不在顶层导出**，从 `feature_derivative.sandbox` 导入）。除以零返回 `None`。三层防护：AST 白名单校验 + `__builtins__` 置空 + locals 仅含所需变量。

#### `BaseStrategy` 抽象基类

- `compute(data, expression, target_col, fill_value=None)` — 抽象方法
- `validate_expression(expression)` — **必须**在 `validate_variables()` 之前调用，否则不安全表达式可能被误报为变量缺失
- `validate_variables(expression, available_vars)` — 校验变量存在性

### 3.5 异常体系

```
Exception
└── FeatureDerivativeError
    ├── MissingVariableError   (missing_vars, available)
    ├── ExpressionSyntaxError
    └── UnsafeExpressionError
```

`MissingVariableError` 构造：`MissingVariableError(missing_vars, available=None)`，公开属性 `missing_vars` / `available`。

### 3.6 AST 白名单

仅允许以下 12 种节点（`+ - * /` 与括号）：

`ast.Expression`、`ast.Load`、`ast.BinOp`、`ast.UnaryOp`、`ast.Add`、`ast.Sub`、`ast.Mult`、`ast.Div`、`ast.USub`、`ast.UAdd`、`ast.Name`、`ast.Constant`

被拒绝的典型节点（触发 `UnsafeExpressionError`）：`ast.Call`（函数调用）、`ast.Attribute`（属性访问）、`ast.Subscript`（下标访问）。可拦截的攻击向量包括 `__import__('os')`、`eval(...)`、`(1).__class__.__bases__...` 等。

---

## 4. risk_ml — 风控建模框架

### 4.1 包结构与导出

**根包导出**（`from risk_ml import ...`）：

| 类别 | 符号 |
|---|---|
| 基类 / 流水线 | `RiskTransformer` / `RiskSelector` / `RiskPipeline` |
| 预处理 | `FeatureCleaner` |
| 分箱 | `ChiMergeBinner` |
| 编码 | `WoeEncoder` / `BinnerWoeEncoder` |
| 特征筛选 | `IVSelector` / `CorrelationSelector` / `PSISelector` |
| 估计器 | `RiskXGBClassifier` / `OptunaTuner` |
| 实验模块 | `TimeWindow` / `ExperimentConfig` / `ExperimentResult` / `ExperimentRunner` / `make_experiment_grid` |
| 指标 | `BaseMetric` / `AUCMetric` / `KSMetric` / `LiftMetric` / `DEFAULT_METRICS` |
| 数据集 | `LendingClubLoader` |

**懒加载兼容导出**：`from risk_ml import ModelReport, ReportContext, ExcelWriter, ...` 等 14 个 report 符号（经 `__getattr__` 从 `risk_report` 重导出），保证旧代码可用。

**注意**：`make_feature_grid` 需 `from risk_ml.experiment import make_feature_grid`（根包不导出）。

### 4.2 基类体系

#### `RiskTransformer(BaseEstimator, TransformerMixin)`

所有 Transformer 的基类。子类实现 `fit(X, y=None) -> self` 与 `transform(X) -> DataFrame`。自动提供 `fit_transform()`、`get_params()`/`set_params()`、pandas 输出（`set_output(transform="pandas")`）。允许 NaN 输入（`_more_tags` 返回 `{"allow_nan": True}`）。

fit 后属性：`feature_names_in_`（list[str]）、`n_features_in_`（int）。

#### `RiskSelector(BaseEstimator, SelectorMixin)`

筛选器基类。子类实现 `fit()` 与 `_get_support_mask() -> np.ndarray`（布尔掩码，True 保留）。自动提供 `transform()`（返回 DataFrame，保留列名）、`get_support()`、`inverse_transform()`、`get_feature_names_out()`。

#### `RiskPipeline(Pipeline)`

扩展 sklearn `Pipeline`，支持验证集数据流与 step 间属性传递。

```python
fit(X, y=None, X_val=None, y_val=None, **fit_params)
```

- 传入 `X_val`/`y_val` 时，每个 transformer step 同步维护验证集数据流
- `PSISelector` 用 `transform(X_val)` 计算真实 PSI 再过滤训练集
- step 间自动把 `iv_values_` 注入后续 `CorrelationSelector(iv_values=None)`
- 最终估计器为 `OptunaTuner` 且有验证集时自动切到 holdout 模式
- `**fit_params` 格式：`step__param=value`

fit 后属性：`X_val_transformed_`（验证集变换后特征）、`y_val_`（验证集标签）。

#### `validate_dataframe(X, reset=False)`

模块级校验函数（`from risk_ml._base import`）。非 DataFrame 抛 `TypeError`，无列抛 `ValueError`。

#### `map_sentinels_to_nan(X, sentinels=None)`

哨兵值映射为 NaN（`from risk_ml._config import`）。默认哨兵 `MISSING_VALUE_SENTINELS = [-999, -9998, -9996]`（-999 一般缺失，-9998 拒绝披露，-9996 系统默认值）。返回副本。

### 4.3 预处理 — `FeatureCleaner`

```python
FeatureCleaner(
    sentinels=None,               # 哨兵列表；None 用默认 [-999,-9998,-9996]；[] 禁用
    missing_threshold=0.95,       # 缺失率 ≥ 此值的列被删除
    missing_strategy="median",    # 'median'/'mean'/'constant'/'drop_row'
    missing_fill_value=None,      # 'constant' 时的填充值
    outlier_method=None,          # None/'iqr'/'percentile'
    outlier_bounds=(0.01, 0.99),  # 'percentile' 上下界分位数
    outlier_iqr_factor=1.5,       # 'iqr' 的 IQR 倍数
    outlier_action="clip",        # 'clip' 截断 / 'set_nan' 设为 NaN
    variance_threshold=0.0,       # 方差 ≤ 此值的列被删除
    nunique_threshold=1,          # 唯一值 ≤ 此值的常数列被删除
)
```

| 方法 | 说明 | 返回 |
|---|---|---|
| `fit(X, y=None)` | 学习填充值、异常值边界、待删除列（`y` 忽略） | `self` |
| `transform(X)` | 哨兵映射 → 删列 → 填充 → 异常值处理 | DataFrame |

fit 后属性：`drop_columns_`（list[str]）、`impute_values_`（dict，`{col: value}`）、`clip_bounds_`（dict，`{col: (lower, upper)}`）。

**异常**：`missing_strategy` / `outlier_method` / `outlier_action` 非法值抛 `ValueError`。

### 4.4 分箱 — `ChiMergeBinner`

```python
ChiMergeBinner(
    max_bins=10,               # 最大分箱数
    min_bins=2,                # 最小分箱数
    bin_pct_threshold=0.05,    # 最小箱占比，低于则合并到相邻箱
    confidence_level=0.9,      # 卡方检验置信度
    special_values=None,       # {col: [特殊值]}，强制独立成箱
    categorical_features=None, # 分类特征列名列表
)
```

| 方法 | 说明 | 返回 |
|---|---|---|
| `fit(X, y=None)` | 卡方自底向上合并分箱（需要 y） | `self` |
| `transform(X)` | 映射为 0-based 整数箱索引 | DataFrame |
| `get_bin_table(feature)` | 分箱汇总表（bin_index/bin_label/bin_lower/bin_upper） | DataFrame |

fit 后属性：`bin_edges_`（dict，`{col: np.ndarray}`）、`bin_labels_`（dict，`{col: list[str]}`）。

**特殊值**：`special_values={"col": [-999]}` 将指定值强制独立成箱，不参与卡方合并，不计入 `max_bins` 约束；每个特殊值一个箱，位于其数值位置的独立区间（中点隔离法）。分类列同样支持（`special_values` 传类别值）。

**异常**：`y is None` 抛 `ValueError`；输入非 DataFrame 抛 `TypeError`。

### 4.5 编码 — `WoeEncoder` / `BinnerWoeEncoder`

#### `WoeEncoder(binner=None, eps=0.001)`

| 方法 | 说明 | 返回 |
|---|---|---|
| `fit(X, y=None)` | 计算各列各箱 WOE 值与 IV（`binner` 非空时先分箱） | `self` |
| `transform(X)` | 箱索引替换为 WOE 值 | DataFrame（浮点） |
| `get_woe_table(feature)` | WOE 明细表（bin_index/woe） | DataFrame |

fit 后属性：`woe_map_`（dict，`{col: {bin_idx: woe}}`）、`iv_values_`（dict）。

**公式**：`WOE(bin) = ln(dist_pos/dist_neg)`，`IV(feature) = Σ(dist_pos - dist_neg) × WOE`。

#### `BinnerWoeEncoder(max_bins=10, min_bins=2, bin_pct_threshold=0.05, confidence_level=0.9, special_values=None, categorical_features=None, eps=0.001)`

分箱 + WOE 一步到位。fit 后属性：`binner_`、`encoder_`、`bin_edges_`、`bin_labels_`、`woe_map_`、`iv_values_`。提供 `get_bin_table(feature)` / `get_woe_table(feature)`。

### 4.6 特征筛选

#### `IVSelector(iv_threshold=0.02, max_iv=0.5, eps=0.001)`

按信息值筛选。保留条件 `iv_threshold <= IV <= max_iv`（低于阈值无预测能力，高于阈值疑似泄露）。fit 后属性：`iv_values_`（pd.Series）。建议输入已 WOE 编码数据。

**IV 参考**：<0.02 无预测能力；0.02-0.1 弱；0.1-0.3 中；>0.3 强（可能泄露）。

#### `CorrelationSelector(corr_threshold=0.7, iv_values=None, strategy="drop_one", max_samples=10000)`

移除高相关对中 IV 较低者。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `corr_threshold` | `0.7` | Pearson 相关系数绝对值阈值 |
| `iv_values` | `None` | 预计算 IV（高相关对的保留决策）；None 用方差替代 |
| `strategy` | `'drop_one'` | `'drop_one'` 保留 IV 高者 / `'drop_both'` 两者都删 |
| `max_samples` | `10000` | 相关矩阵最大样本量，超采样（random_state=42）；0/None 禁用采样 |

静态方法（无需构造实例）：

```python
CorrelationSelector.compute_correlation_matrix(X, max_samples=10000)  # -> pd.DataFrame
CorrelationSelector.compute_high_corr_pairs(X, threshold=0.7, max_samples=10000)  # -> list[tuple[str, str, float]]
```

fit 后属性：`correlation_matrix_`（pd.DataFrame）、`drop_features_`（list[str]）。

#### `PSISelector(psi_threshold=0.25, n_bins=10, eps=1e-4)`

按群体稳定性指数筛选。`fit` 记录训练集参考分布（`reference_dist_`），`transform` 时计算 `psi_values_` 并筛选（`PSI <= psi_threshold` 保留）。

**PSI 参考**：<0.1 稳定；0.1-0.25 边际；>0.25 不稳定。

### 4.7 估计器

#### `RiskXGBClassifier(...)`

风控场景 XGBoost 二分类器，封装 `xgboost.XGBClassifier`，完全 sklearn 兼容。

```python
RiskXGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    scale_pos_weight=1, min_child_weight=5,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, eval_metric="auc",
    tree_method="hist", n_jobs=-1, **xgb_kwargs,
)
```

| 方法 | 说明 | 返回 |
|---|---|---|
| `fit(X, y, **fit_kwargs)` | 拟合；DataFrame 输入记录特征名，object 列自动转 category（原生分类支持）；`**fit_kwargs` 如 `eval_set`、`early_stopping_rounds` | `self` |
| `predict(X)` | 类别标签 | `np.ndarray` |
| `predict_proba(X)` | 类别概率，形状 `(n_samples, 2)` | `np.ndarray` |
| `predict_score(X)` | 正例概率（风控评分） | `np.ndarray` |
| `feature_importance(importance_type="gain")` | 特征重要性 dict；类型 `weight/gain/cover/total_gain/total_cover` | `dict` |

fit 后属性：`model_`（底层 XGBClassifier）、`classes_`、`feature_names_in_`。

**风控调参建议**：`max_depth` 3~5；`learning_rate` 0.01~0.1；`min_child_weight` 偏大；`scale_pos_weight = 负样本数/正样本数`。

#### `OptunaTuner(estimator, n_trials=50, search_space=None, scoring="roc_auc", cv=5, n_jobs=1, random_state=42, early_stopping_rounds=10, verbose=0)`

基于 Optuna TPE 的贝叶斯超参搜索器，兼容任意 sklearn 分类器。

- `search_space`：`{参数名: (low, high)}`，数值参数自动识别 int/float；None 用风控默认空间
- `fit(X, y, X_val=None, y_val=None, **fit_params)`：两种模式 — CV 模式（默认，cross_val_score）/ Holdout 模式（传 `X_val`/`y_val`）
- `scoring` 支持 `"roc_auc"/"f1"/"ks"/"accuracy"` 等

fit 后属性：`best_params_`、`best_score_`、`best_estimator_`（最优参数全量重训）、`study_`（optuna.Study）、`trials_dataframe_`。

提供 `predict` / `predict_proba` / `predict_score`。

### 4.8 实验模块

#### `TimeWindow(date_column, start_date, end_date)`（dataclass）

时间窗口配置。`filter(X) -> pd.Series` 返回满足 `start <= date <= end` 的布尔掩码。`__str__` 返回 `"start~end"`。

#### `ExperimentConfig(name, label_col, time_window=None, weight_col=None, feature_columns=None, pipeline=None, fit_kwargs=None)`（dataclass）

单次实验配置。`feature_columns` 可指定不同特征组合（优先于 runner 级）；`pipeline` 支持自定义；`fit_kwargs` 传给流水线 fit（Pipeline 时 `sample_weight` 路由为 `{最后一步}__sample_weight`）。

#### `ExperimentResult`（dataclass）

单次实验结果。关键字段：`status`（success/failed）、`error`、`estimator`（拟合后最优流水线）、`best_params`、`n_samples`、`n_features`、`default_rate`、`mean_iv`、`metric_values`、`oot_metric_values`、`training_time` 等。

#### 配置生成器

```python
make_experiment_grid(label_cols, time_windows=None, weight_cols=None, name_prefix="exp")
# 标签×时间窗口×权重的笛卡尔积 -> list[ExperimentConfig]

make_feature_grid(feature_groups, label_col, time_window=None, weight_col=None, name_prefix="feat")
# 每组特征一个实验，用于比较不同特征组合
```

#### 指标体系

`BaseMetric` 抽象基类：实现 `name` 属性与 `compute(y_true, y_score) -> float`。

```python
class GiniMetric(BaseMetric):
    name = "gini"
    def compute(self, y_true, y_score):
        from sklearn.metrics import roc_auc_score
        return 2 * roc_auc_score(y_true, y_score) - 1
```

内置指标：`AUCMetric()`（auc）、`KSMetric()`（ks）、`LiftMetric(percentile=10)`（lift_{pct}）。`DEFAULT_METRICS = [AUCMetric(), KSMetric(), LiftMetric(10)]`。

#### `ExperimentRunner(...)`

```python
ExperimentRunner(
    configs,                  # list[ExperimentConfig] 必填
    pipeline=None,            # 默认流水线；None 用标准流水线
    feature_columns=None,     # 特征列；None 自动推断
    metrics=None,             # 指标；None 用 DEFAULT_METRICS
    scoring="ks",             # Optuna 优化目标 + best 选举依据
    n_trials=30,              # 每实验 Optuna 轮数
    tuner_cv=3,               # Optuna 内部 CV 折数
    oot=None,                 # OOT 跨时间验证数据集
    eval_label_cols=None,     # 额外评估标签列
    n_jobs=1, random_state=42, verbose=1,
)
```

默认流水线：`FeatureCleaner → BinnerWoeEncoder → IVSelector → CorrelationSelector → RiskXGBClassifier`。

| 方法 | 说明 | 返回 |
|---|---|---|
| `fit(X, y=None)` | 运行所有实验（`y` 忽略，标签从 config.label_col 提取） | `self` |
| `predict(X)` / `predict_proba(X)` / `predict_score(X)` | 用最优估计器预测 | `np.ndarray` |
| `show(top_n_features=10)` | 生成 **7 节 Markdown 实验报告** | `str` |

fit 后属性：
- `experiments_`（dict，`{name: ExperimentResult}`）
- `results_`（DataFrame，每行一个实验，含训练/OOT/额外标签指标列）
- `best_config_` / `best_estimator_` / `best_score_`

**`show()` 的 7 节内容**：实验概览 → 指标对比表（拆 5 个子表防宽表）→ 训练→OOT 衰减分析（>20% 标 ⚠️）→ Top-N 特征重要性对比 → 特征稳定性分析 → 最优超参数对比 → 最优实验详情。

**异常**：所有实验失败时 `fit` 抛 `RuntimeError`；未 fit 调用 `predict`/`show` 抛 `RuntimeError`。

### 4.9 数据集 — `LendingClubLoader`

```python
LendingClubLoader(
    data_dir="~/.risk_ml/datasets/lending_club",
    use_features="selected",  # 'selected' 精选 ~30 列 / 'all' / 自定义 set
    sample_ratio=None,        # 采样比例
    random_state=42,
    drop_leakage=True,        # 自动剔除泄露特征
)
```

| 方法 | 说明 | 返回 |
|---|---|---|
| `load()` | 下载/缓存/预处理（目标二值化，违约=1） | `(X, y)` |
| `data_dictionary()` | 精选特征中文数据字典 | `dict` |

load 后属性：`n_samples_`、`n_features_`、`feature_names_`、`default_rate_`。

---

## 5. risk_report — 模型报告模块

### 5.1 三种使用模式

```python
from risk_report import (
    ModelReport, ReportContext, ExcelWriter,
    ScoreLiftOperator, compute_lift_table,
    DEFAULT_DOCUMENT_CONFIG, DocumentConfig, SheetConfig,
)

# 1. 日常单独调用（无需构造上下文）
df = ScoreLiftOperator.compute_lift_table(y_true, y_score, n_bins=10)

# 2. 模块化组装（自定义配置）
config = DocumentConfig(sheets=[
    SheetConfig("模型表现", [ScoreLiftOperator(), ModelEffectOperator()]),
])
report = ModelReport(config=config).fit(context).to_excel("report.xlsx")

# 3. 全量报告（默认 8 Sheet / 22 算子）
context = ReportContext(data=df, tag_col="tag", label_col="is_fraud",
                        pipeline=fitted_pipeline, time_col="transaction_time")
ModelReport().fit(context).to_excel("report.xlsx")
```

### 5.2 数据输入 — `ReportContext`

单 DataFrame + tag 列，替代传统 X_train/y_train/X_test 多数组。构造时自动执行：流水线属性提取、预测分数计算、IV 自动计算。

```python
context = ReportContext(
    data=df,                       # 完整 DataFrame（含 tag/label/score 列）
    tag_col="tag",                 # 区分 train/test/oot
    label_col="is_fraud",          # 主标签列
    extra_labels=["y_mob3", "y_mob6"],  # 额外标签列（MOB 压测）
    pipeline=pipe,                 # 已拟合流水线（自动提取属性与预测分数）
    score_col="score",             # 模型分数列（可由 pipeline 自动计算）
    baseline_score_col="baseline_score",  # 对标模型分数列
    gray_tag=None,                 # 灰样本 tag 值
    time_col="transaction_time",   # 时间列（月度拆分分析）
    model_name="", developer="", validator="", business_owner="",
    background="", application="", observation_period="",
    label_definition={0: "好", -1: "灰", 1: "坏"},
    metrics=None,                  # 默认 DEFAULT_METRICS
    sample_origin_distribution=None,  # 原始样本分布（无时自动计算）
    sub_models=None,               # {"征信子": {"score_col": ..., "label_col": ...}}
    portrait_data=None,            # 画像数据
    feature_meta=None,             # {col: {"含义": ..., "来源": ..., "类别": ...}}
)
```

**辅助方法**：

| 方法 | 返回 |
|---|---|
| `get_datasets(label_col=None)` | `{"训练集": (y_true, y_score), "测试集": ..., "跨时间验证集": ...}` |
| `get_baseline_datasets(label_col=None)` | `{"训练集": (y_true, baseline_score), ...}` |
| `get_gray_datasets(label_col=None)` | `{"灰样本": (y_true, y_score)}` 或 `{}` |
| `get_datasets_with_gray(label_col=None)` | 含灰样本的数据集（y_true 含 -1） |
| `get_sample_stats(label_col=None)` | 各数据集好坏灰数量与坏占比 |
| `get_monthly_datasets(label_col=None, tag_val="oot")` | 按月拆分 `{"2026-04": (y_true, y_score)}` |
| `get_monthly_baseline_datasets(label_col=None, tag_val=None)` | 按月拆分对标分数 |

**`PipelineAttributes`（dataclass）**：从已拟合流水线提取的属性汇总（`drop_columns_`、`woe_map_`、`iv_values_`、`bin_edges_`、`correlation_matrix_`、`feature_importance_gain_`、`model_params_` 等，字段可为 None）。`extract_pipeline_attributes(pipeline)` 提取，`ReportContext.pipeline_attrs` 为自动填充的公开属性。`TAG_CN_MAP = {"train": "训练集", "test": "测试集", "oot": "跨时间验证集"}`。

### 5.3 组合器 — `ModelReport`

```python
ModelReport(config: DocumentConfig | None = None)   # config=None 用默认 8 Sheet/22 算子模板
```

| 方法 | 说明 | 返回 |
|---|---|---|
| `fit(context: ReportContext)` | 按 config.sheets 顺序运行算子 | `self` |
| `to_excel(file_path, format_config=None)` | 写入 Excel（同 sheet 算子垂直排列） | None |
| `get_result(name)` | 取指定算子结果；不存在抛 `KeyError` | `list[SubSection]` |

fit 后属性：`results_`（dict，`{算子name: list[SubSection]}`）、`context`。

**配置类**：

```python
SheetConfig(sheet_name, operators)        # 一个 Sheet 与算子列表
DocumentConfig(sheets, format_config=None)  # 整份文档配置
```

### 5.4 8 个 Sheet / 22 算子

| Sheet | 算子 |
|---|---|
| 模型说明 | MetaInfoOperator |
| 1.模型设计 | DevPurposeOperator, ModelAssumptionOperator, LabelDefinitionOperator, SampleSelectionOperator, ModelingSampleOperator, EffectSummaryOperator |
| 2.变量分析 | VarDescriptionOperator, VarCleaningOperator, VarFilterOperator, VarAnalysisOperator |
| 附件-变量分箱 | VarBinningOperator |
| 3.模型表现 | ModelMethodOperator, ModelEffectOperator, ScoreLiftOperator, ScoreLiftGrayOperator |
| 附件1-补充分析 | AttributionOperator, ModelComparisonOperator, MobPerformanceOperator, PortraitOperator |
| 附件2-模型使用方案 | SwapAnalysisOperator |
| 附件3-变量描述 | VarRangeOperator |

**算子基类契约**：`name`（英文名）、`title`（中文标题）、`compute(context) -> list[SubSection]`。`SubSection(title, data, note="")` 是 sheet 内一个独立分析块。数据缺失时算子产出 `placeholder_df(msg)` 占位表保证模板结构完整。

**带构造参数的算子**：

| 算子 | 构造参数 | 签名默认 | 生效默认（None 时） |
|---|---|---|---|
| `EffectSummaryOperator` / `ModelComparisonOperator` | `lift_percentiles` | `None` | `[10]` |
| `ModelEffectOperator` | `lift_percentiles` | `None` | `[10, 5, 2, 1]` |
| `ScoreLiftOperator` / `ScoreLiftGrayOperator` / `VarBinningOperator` | `n_bins` | `10` | `10` |
| `SwapAnalysisOperator` | `cutoff_percentiles` | `None` | `[10, 20]` |

**公开静态方法**（日常单独调用）：
- `ScoreLiftOperator.compute_lift_table(y_true, y_score, n_bins=10, baseline_score=None)`
- `ModelEffectOperator.compute_effect_table(datasets, metrics=None, lift_percentiles=[10,5,2,1])`
- `SwapAnalysisOperator.compute_swap_table(y_true, y_score_new, y_score_old=None, cutoff_percentiles=[10,20])`

### 5.5 独立计算函数（`risk_report._scoring`）

这些函数同时被 `risk_ml` 的 WoeEncoder / IVSelector / CorrelationSelector 委托调用（统一算法）。

```python
compute_lift_table(y_true, y_score, n_bins=10, baseline_score=None) -> pd.DataFrame
# 列: min/max/goods/bads/total/total%/bad_rate/ks/lift/cum_lift；有 baseline 时加 baseline_ 前缀列

compute_swap_analysis(y_true, y_score_new, y_score_old=None, cutoff_percentiles=[10, 20]) -> pd.DataFrame
# Swap In/Out 分析，列: 切分比例/新总拒绝/新通过/新拒绝坏人/新通过好人；有对标时加 swap_in/swap_out/净改善

compute_per_feature_ks(X, y_true, y_score, n_bins=10) -> pd.Series
# 每个特征单特征区分度 KS，索引为特征名

compute_sample_stats(y, label_definition=None) -> dict
# 键: goods/bads/gray/total/total_with_gray/bad_rate

compute_iv_from_data(X, y, eps=0.001) -> pd.Series
# 统一 IV 算法，numba JIT+并行（未装 numba 回退 bincount）

compute_correlation_matrix(X, max_samples=10000, method="pearson") -> pd.DataFrame
# 特征相关矩阵，采样加速；method 目前仅支持 "pearson"

compute_high_corr_pairs(X, threshold=0.7, max_samples=10000) -> list[tuple[str, str, float]]
# 高相关特征对列表，(特征A, 特征B, 相关系数绝对值)
```

### 5.6 Excel 输出 — `ExcelWriter` / `FormatConfig`

```python
ExcelWriter(file_path, format_config=None, auto_adjust_width=True)
# 方法: write_sub_sections(sub_sections, sheet_name, format_config=None)
#       write_dataframe(df, sheet_name, start_row=1, format_config=None, title=None)
#       write_report(sheet_results)  /  save()
```

`FormatConfig` 默认值：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `font_name` | `"微软雅黑"` | 字体 |
| `alt_row_color` / `alt_row_enabled` | `"F2F7FB"` / `True` | 交替行色 |
| `data_bar_columns` | `["lift","cum_lift","bad_rate","gain_per","weight_per"]` | 数据条列 |
| `percent_columns` | `["bad_rate","total%","坏占比","缺失率"]` | 百分比格式列 |
| `freeze_header_sheets` | `["2.变量分析","附件3-变量描述"]` | 冻结表头 Sheet |
| `header_bg_color` | `"4472C4"` | 表头底色 |

---

## 6. 端到端实战示例

### 6.1 特征衍生 → 建模 → 报告

```python
import pandas as pd
from feature_derivative import transform
from risk_ml import (
    FeatureCleaner, BinnerWoeEncoder, IVSelector,
    CorrelationSelector, RiskXGBClassifier, RiskPipeline,
)
from risk_report import ModelReport, ReportContext

# ========== 1. 特征衍生 ==========
df = pd.DataFrame({"age": [30, 40, 25], "income": [10, 8, 5],
                   "balance": [5, 6, 3], "is_fraud": [0, 1, 0],
                   "tag": ["train"]*3, "t": ["2026-01","2026-02","2026-03"]})
df = transform(df, "income/age", "income_per_age")

# ========== 2. 建模流水线 ==========
pipe = RiskPipeline([
    ("cleaner", FeatureCleaner(missing_strategy="median")),
    ("binner_woe", BinnerWoeEncoder(special_values={"income": [-999]})),
    ("iv_selector", IVSelector(iv_threshold=0.02)),
    ("corr_selector", CorrelationSelector(corr_threshold=0.7)),
    ("clf", RiskXGBClassifier()),
])

X = df[["age", "income", "balance", "income_per_age"]]
y = df["is_fraud"]
pipe.fit(X, y)
y_score = pipe.predict_score(X)         # 默认=正例概率；配置 score_scaler 后为拉伸风险分

# ========== 3. 自动产出报告 ==========
context = ReportContext(
    data=df.assign(score=y_score),
    tag_col="tag", label_col="is_fraud",
    pipeline=pipe, time_col="t",
)
ModelReport().fit(context).to_excel("model_report.xlsx")
```

### 6.2 实验对比 + Markdown 报告

```python
from risk_ml.experiment import (
    ExperimentRunner, ExperimentConfig, make_feature_grid,
    TimeWindow, AUCMetric, KSMetric, LiftMetric,
)

configs = make_feature_grid(
    feature_groups=[["age", "income"], ["age", "income", "balance"]],
    label_col="is_fraud",
    time_window=TimeWindow("t", "2026-01-01", "2026-12-31"),
)

runner = ExperimentRunner(
    configs=configs,
    metrics=[AUCMetric(), KSMetric(), LiftMetric(10)],
    scoring="ks",
    n_trials=20,
    oot=df_oot,                    # OOT 跨时间验证
    eval_label_cols=["y_mob6"],    # 多标签评估
)
runner.fit(df)
print(runner.show(top_n_features=10))   # 7 节 Markdown 报告
y_score = runner.predict_score(X_test)
```

---

## 7. 附录

### 7.1 运行测试

```bash
python -m pytest tests/ risk_ml/tests/ risk_report/tests/ -q --tb=short
```

### 7.2 sklearn 兼容约定

- 所有算子继承 sklearn 基类，`fit`/`transform`/`predict` 接口标准
- `get_params()` / `set_params()` / `clone()` 可用（支持 GridSearchCV）
- fit 后 `_` 结尾属性为公开属性
- Transformer 默认 pandas 输出；`RiskSelector.transform` 返回 DataFrame 并保留列名
- 允许 NaN 输入（`_more_tags` 返回 `{"allow_nan": True}`）

### 7.3 已知限制与注意事项

- `feature_derivative` 表达式仅支持四则运算与括号，不支持函数调用、比较运算
- `UnsafeExpressionError` 仅在 Online（dict）模式触发；Pandas/Spark 模式向量化无此风险
- `make_feature_grid` 需从 `risk_ml.experiment` 导入
- `from risk_ml.report import X` 不再支持（子模块已迁移为独立包）；请用 `from risk_ml import X` 或 `from risk_report import X`
- `compute_correlation_matrix` 的 `method` 参数目前仅支持 `"pearson"`
- `LendingClubLoader` 首次下载约 1.1 GB（Kaggle），需安装 `kaggle` 包并配置凭据

### 7.4 性能说明

| 计算 | 优化手段 | 指标 |
|---|---|---|
| IV 计算 | numba JIT + 并行，bincount 回退 | 200 列 × 10 万行 ≈ 0.3s |
| 相关性计算 | 采样 10000 行 + numpy corrcoef | 200 列 × 10 万行 ≈ 0.04s（203x） |
| numba warmup | 模块导入 0s，首次调用 IV 时延迟 JIT 编译 | — |

---

## 8. 进阶开发指南

本章面向需要扩展框架的进阶用户，给出三个实战主题：

- **8.1 自定义 risk_ml 算子** — 继承 `RiskTransformer` / `RiskSelector` 编写新转换器与筛选器，并接入标准流水线
- **8.2 自定义 report 算子** — 继承 `ReportOperator` 编写新分析块，注册进 `DocumentConfig` 产出到 Excel
- **8.3 独立训练的子算子拼装 pipeline 生成报告** — 各自单独训练的子算子组装成 pipeline，一键产出标准模型报告

以下示例统一使用一组合成数据（§6.1 的 `demo_data.csv` 同样适用）：

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 3000
df = pd.DataFrame({f"f{i}": np.random.randn(n) for i in range(12)})  # 12 个噪声特征
df["amount"] = np.random.lognormal(mean=3, sigma=0.8, size=n)        # 弱偏态金额
df["days_since"] = np.random.randint(0, 400, size=n)
df["age"] = np.random.randint(18, 70, size=n)

# 坏样本率约 9%：log金额 / 距上次交易天数 / 年龄 / f0 / f1 是信号
logit = (-3.0 + 0.8 * (np.log(df["amount"]) - 3) + 0.005 * (df["days_since"] - 200)
         + 0.03 * (df["age"] - 40) + 0.6 * df["f0"] - 0.5 * df["f1"])
df["is_fraud"] = (np.random.rand(n) < 1 / (1 + np.exp(-logit))).astype(int)

# tag 划分: train 50% / test 30% / oot 20%
df["tag"] = "train"
idx = df.index.to_numpy()
perm = np.random.permutation(idx)
df.loc[perm[:int(n * 0.2)], "tag"] = "oot"
df.loc[perm[int(n * 0.2):int(n * 0.5)], "tag"] = "test"

feature_cols = [c for c in df.columns if c not in {"is_fraud", "tag"}]
X_train, y_train = df[df["tag"] == "train"][feature_cols], df.loc[df["tag"] == "train", "is_fraud"]
X_test,  y_test  = df[df["tag"] == "test"][feature_cols],  df.loc[df["tag"] == "test", "is_fraud"]
```

### 8.1 自定义 risk_ml 算子

自定义入口是 `risk_ml._base` 的两个基类（详见 §4.2），均继承 sklearn 基类：

| 基类 | 必须实现 | 自动获得 |
|---|---|---|
| `RiskTransformer` | `fit(X, y=None) → self`、`transform(X) → DataFrame` | `fit_transform`、`get_params/set_params`、pandas 输出、`allow_nan` 标签 |
| `RiskSelector` | `fit(X, y=None) → self`、`_get_support_mask() → np.ndarray` | `transform`（按掩码筛列）、`get_support`、`inverse_transform`、`get_feature_names_out` |

约定（见 §7.2）：fit 后以 `_` 结尾的属性为公开属性；基类 `fit` 已负责 `validate_dataframe` 校验并记录 `feature_names_in_` / `n_features_in_`，子类可用 `super().fit(X, y)` 复用。

**示例：极值缩尾转换器**

```python
from risk_ml import RiskTransformer
from risk_ml._base import validate_dataframe


class WinsorizeTransformer(RiskTransformer):
    """将每列超上下分位数的值裁剪到分位数处（缩尾）。"""

    def __init__(self, lower_quantile=0.01, upper_quantile=0.99, columns=None):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.columns = columns

    def fit(self, X, y=None):
        super().fit(X, y)  # 校验 + 记录 feature_names_in_ / n_features_in_
        cols = self.columns or X.select_dtypes(include="number").columns.tolist()
        self.columns_ = [c for c in cols if c in X.columns]
        self.lowers_ = {c: float(X[c].quantile(self.lower_quantile)) for c in self.columns_}
        self.uppers_ = {c: float(X[c].quantile(self.upper_quantile)) for c in self.columns_}
        return self

    def transform(self, X):
        validate_dataframe(X)
        out = X.copy()
        for c in self.columns_:
            out[c] = out[c].clip(lower=self.lowers_[c], upper=self.uppers_[c])
        return out
```

**示例：方差筛选器**

```python
from risk_ml import RiskSelector


class VarianceSelector(RiskSelector):
    """按方差阈值筛选特征（非数值列默认保留）。"""

    def __init__(self, threshold=0.005):
        self.threshold = threshold

    def fit(self, X, y=None):
        super().fit(X, y)
        self.variances_ = X.var(numeric_only=True)  # 只对数值列算方差
        return self

    def _get_support_mask(self):
        # 掩码长度必须与 feature_names_in_ 对齐
        mask = []
        for col in self.feature_names_in_:
            if col in self.variances_.index:
                mask.append(bool(self.variances_[col] >= self.threshold))
            else:
                mask.append(True)  # 非数值列默认保留
        return np.array(mask)
```

**独立使用**

```python
winsor = WinsorizeTransformer(columns=["amount", "days_since"]).fit(X_train)
X_w = winsor.transform(X_train)          # 或直接 fit_transform(X_train)

sel = VarianceSelector(threshold=0.5).fit(X_train)
sel.get_support()                        # np.ndarray 布尔掩码
X_sel = sel.transform(X_train)           # 自动按掩码筛选列
```

**接入标准流水线**

```python
from risk_ml import (FeatureCleaner, BinnerWoeEncoder, IVSelector,
                     RiskXGBClassifier, RiskPipeline)

pipe = RiskPipeline([
    ("winsor", WinsorizeTransformer()),            # 自定义转换器
    ("cleaner", FeatureCleaner(missing_threshold=0.9, variance_threshold=0.0)),
    ("bwe", BinnerWoeEncoder(max_bins=6)),
    ("iv", IVSelector(iv_threshold=0.02)),
    ("var", VarianceSelector(threshold=0.0)),      # 自定义筛选器
    ("clf", RiskXGBClassifier(n_estimators=100, eval_metric="auc")),
])
pipe.fit(X_train, y_train)
y_score = pipe.predict_proba(X_test)[:, 1]
```

自定义算子完全兼容 sklearn 生态：`get_params()` / `set_params()` / `clone()` 可用于 `GridSearchCV`，也可直接作为 `ExperimentRunner` 的 pipeline 参与自动调参（见 §4.8）。

### 8.2 自定义 report 算子

`ReportOperator` 是一次性计算生产者（无 fit/transform 周期），需实现三个成员：

| 成员 | 说明 |
|---|---|
| `name`（属性） | 算子英文名，如 `"percentile_stability"`，作为 `ModelReport.get_result()` 的键 |
| `title`（属性） | 算子中文标题，如 `"分数分位数稳定性"` |
| `compute(context) → list[SubSection]` | 从 `ReportContext` 读取数据计算，返回一个或多个 `SubSection` |

`SubSection(title, data, note="")` 是 Excel sheet 内的一个分析块；数据缺失时用 `placeholder_df(msg)` 产出占位表，保持模板结构完整。

**示例：分数分位数算子**

```python
from risk_report import ReportOperator, SubSection, placeholder_df


class PercentileStabilityOperator(ReportOperator):
    """各数据集分数分位数（P50/P90/P99）。"""

    name = "percentile_stability"
    title = "分数分位数稳定性"

    def compute(self, context):
        if context.data is None or context.score_col is None:
            return [SubSection(title=self.title,
                               data=placeholder_df("缺少 data / score_col，无法计算分位数"))]
        df = context.data.dropna(subset=[context.score_col])
        rows = []
        for tag_val, cn in [("train", "训练集"), ("test", "测试集"), ("oot", "跨时间验证集")]:
            sub = df[df[context.tag_col] == tag_val]
            if sub.empty:
                continue
            s = sub[context.score_col]
            rows.append({"数据集": cn, "样本量": len(sub),
                         "P50": round(float(s.median()), 4),
                         "P90": round(float(s.quantile(0.9)), 4),
                         "P99": round(float(s.quantile(0.99)), 4)})
        return [SubSection(title=self.title, data=pd.DataFrame(rows))]
```

**注册进 DocumentConfig 并产出报告**

```python
from risk_report import (ReportContext, ModelReport, DocumentConfig,
                         SheetConfig, DEFAULT_DOCUMENT_CONFIG)

# 方式一: 追加一个新 Sheet（保留默认 8 个 sheet）
custom_config = DocumentConfig(
    sheets=DEFAULT_DOCUMENT_CONFIG.sheets + [SheetConfig("9.自定义分析", [PercentileStabilityOperator()])]
)

# 方式二: 往既有 Sheet 追加算子（如追加到「3.模型表现」）
sheets = []
for s in DEFAULT_DOCUMENT_CONFIG.sheets:
    if s.sheet_name == "3.模型表现":
        sheets.append(SheetConfig(s.sheet_name, s.operators + [PercentileStabilityOperator()]))
    else:
        sheets.append(s)
config2 = DocumentConfig(sheets=sheets)

# 需要 data 中已有分数列（或传入 pipeline 自动计算）
df["score"] = pipe.predict_proba(df[feature_cols])[:, 1]   # pipe 见 8.1
context = ReportContext(data=df, tag_col="tag", label_col="is_fraud",
                        score_col="score", pipeline=pipe)

report = ModelReport(config=custom_config)
report.fit(context)
report.get_result("percentile_stability")   # 按算子 name 取结果
report.to_excel("自定义报告.xlsx")            # Excel 将多出「9.自定义分析」sheet
```

> 提示：算子结果与 Sheet 解耦 — `SheetConfig` 只决定算子结果写入哪个 sheet、在什么位置，算子本身不感知 sheet（见 §5.3 / §5.4）。

### 8.3 独立训练的子算子拼装 pipeline 生成报告

**场景**：清洗、分箱、WOE、筛选、建模分步进行（例如共享一份清洗/分箱结果给多个模型，或分步调参后保存了各子算子），最终想把它们组装起来一次性产出标准模型报告。

**关键机制**：报告的 `extract_pipeline_attributes()` 只按**属性名**遍历 pipeline 各 step（`drop_columns_` / `bin_edges_` / `woe_map_` / `iv_values_` / `correlation_matrix_` / `feature_importances_` 等），与 step 名、类名无关。因此**任意已拟合的 sklearn `Pipeline`（甚至单估计器）都能被报告识别**，无需重新训练。

```python
from sklearn.pipeline import Pipeline
from risk_ml import (FeatureCleaner, ChiMergeBinner, WoeEncoder, IVSelector,
                     CorrelationSelector, RiskXGBClassifier)

# ---- 1) 各子算子独立训练（可跨代码块 / 会话复用已保存的算子） ----
cleaner = FeatureCleaner(missing_threshold=0.9, variance_threshold=0.0)
X_c = cleaner.fit_transform(X_train, y_train)              # ① 清洗

binner = ChiMergeBinner(max_bins=6)
X_b = binner.fit_transform(X_c, y_train)                   # ② 卡方分箱

encoder = WoeEncoder()
X_w = encoder.fit_transform(X_b, y_train)                  # ③ WOE 编码

iv_sel = IVSelector(iv_threshold=0.02)
X_i = iv_sel.fit_transform(X_w, y_train)                   # ④ IV 筛选

corr_sel = CorrelationSelector(corr_threshold=0.7, iv_values=iv_sel.iv_values_)
X_f = corr_sel.fit_transform(X_i, y_train)                 # ⑤ 相关性筛选

clf = RiskXGBClassifier(n_estimators=100, eval_metric="auc")
clf.fit(X_f, y_train)                                      # ⑥ 建模

# ---- 2) 拼装成 sklearn Pipeline（不再 fit，仅用于报告属性提取 + 打分） ----
pipe = Pipeline([
    ("cleaner", cleaner),
    ("binner", binner),
    ("encoder", encoder),
    ("iv", iv_sel),
    ("corr", corr_sel),
    ("clf", clf),
])

# ---- 3) 生成报告 ----
from risk_report import ReportContext, ModelReport

context = ReportContext(
    data=df,                # 原始 DataFrame（含原始特征 + tag + label）
    tag_col="tag",
    label_col="is_fraud",
    pipeline=pipe,          # 已拟合的拼装流水线
    model_name="独立训练拼装模型",
)
# 构造时自动完成: ① 提取 pipeline 属性 ② 若 data 无分数列，调用
#   pipe.predict_proba 计算并写入 data['__y_score__']，设置 score_col

report = ModelReport()
report.fit(context)
report.to_excel("独立训练拼装报告.xlsx")
```

**注意事项**

1. **特征列识别**：`ReportContext` 通过 `pipeline.feature_names_in_` 精确识别特征列。拼装后若从未对 `pipe` 调用 `fit`，则退化为"排除 `tag` / `label` / `score` 等列"的兜底逻辑。**若 `data` 还含 ID、时间等非特征列，请在拼装后对 `pipe` 补一次 `fit(X_train, y_train)`**（各 step 已拟合，重复 fit 结果一致），使 `feature_names_in_` 生效，避免非特征列被送入模型。
2. **自动打分**：只要 `data` 未含 `score_col`，构造 `ReportContext` 时就会自动调用 `pipeline.predict_proba(data[特征列])[:, 1]` 写入 `__y_score__`。若想用别的分数（如某次预测结果），预先写入 `data` 并传 `score_col="列名"` 即可。
3. **特征重要性**：`RiskXGBClassifier` 将底层模型封装在 `model_` 内，不直接暴露 `feature_importances_`。`extract_pipeline_attributes` 已内置对 `feature_importance()` 方法（gain / weight）的兜底提取（v1.0 新增），报告「变量分析」会正常展示 gain / weight 列。
4. **独立使用**：拼装后的 `pipe` 本身也可正常 `predict_proba`，与直接训练整个 pipeline 等价。
