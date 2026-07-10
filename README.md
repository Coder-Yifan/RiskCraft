# Feature Derivative Framework（特征衍生框架）

多端兼容的特征衍生框架 — 接收四则运算表达式，自动解析变量，在三种计算引擎上高效生成新特征列。

## 快速开始

```python
from feature_derivative import transform

# Pandas
import pandas as pd
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
result = transform(df, "a/(a+b)", "ratio")

# Dict（在线服务）
result = transform({"a": 1, "b": 4}, "a/(a+b)", "ratio")

# PySpark
# result = transform(spark_df, "a/(a+b)", "ratio")
```

## 三端适配

| 引擎 | 输入类型 | 计算方式 | 适用场景 |
|------|---------|---------|---------|
| **Pandas** | `pandas.DataFrame` | `df.eval()` 向量化 | 离线批处理 |
| **PySpark** | `pyspark.sql.DataFrame` | `F.expr()` 分布式 | 大规模数据 |
| **Online** | `dict` | 安全沙箱 `eval()` | 在线推理服务 |

框架根据输入数据类型**自动选择**引擎，无需手动指定。

## 核心特性

### 1. 自动引擎识别

```python
transform(data, expression, target_col)  # 自动识别 data 类型
```

### 2. 变量校验

表达式变量缺失时抛出 `MissingVariableError`，包含缺失字段名和可用字段列表。

### 3. 安全沙箱（Online 模式）

三层防护机制：
1. **AST 节点白名单** — 仅允许四则运算节点，拒绝函数调用、属性访问等
2. **`__builtins__` 置空** — 切断所有内置函数
3. **locals 限定** — 只传入表达式所需的变量

### 4. 缺失值处理

支持两种模式（通过 `fill_value` 参数切换）：

- **传播模式**（默认 `fill_value=None`）：NaN/None 参与运算 → 结果为 NaN/None
- **预填充模式**（`fill_value=0` 等）：计算前将 NaN/None 替换为指定值

---

## 缺失值处理策略总结

### 三端默认行为对比

| 场景 | Pandas | PySpark | Online |
|------|--------|---------|--------|
| 输入含 NaN/None | NaN 传播 | null 传播 | None 传播（TypeError → None） |
| 除以零 | inf → NaN | null | ZeroDivisionError → None |
| 预填充模式 | `fillna(fill_value)` | `fillna({col: fill_value})` | None → fill_value |

### 为什么三端选择不同的底层实现？

**1. Pandas — NaN 传播 + inf → NaN**

Pandas 底层基于 NumPy，NaN 传播是 NumPy 的原生行为：任何 NaN 参与的运算结果为 NaN。这符合统计学原则——"缺失输入产生缺失输出"，不会引入虚假数据。除以零时 NumPy 产生 `inf`，我们统一替换为 `NaN`，避免下游出现 `inf` 污染。

**2. PySpark — null 传播（ANSI SQL 标准）**

Spark SQL 遵循 ANSI SQL 标准：null 参与运算结果为 null。这是分布式计算领域的事实标准，无需额外处理。除以零在非 ANSI 模式下（`spark.sql.ansi.enabled=false`，即默认配置）返回 null，与 null 传播策略天然一致。如果启用 ANSI 模式，除以零会抛出异常，需要额外处理——因此我们建议关闭 ANSI 模式。

**3. Online — None 传播（TypeError 捕获）**

Python 原生中 None 参与算术运算会抛出 TypeError。我们捕获该异常并返回 None，实现与 Pandas/PySpark 一致的"缺失传播"语义。除以零捕获 ZeroDivisionError 返回 None，作为"无法计算"的明确信号。

### 统一语义

虽然三端的底层实现不同，但**对外语义是一致的**：

> **缺失输入 → 缺失输出**

这意味着调用方无需关心底层引擎，无论使用哪种数据类型，缺失值的行为都是可预测的。

### fill_value 的适用场景

在以下场景中，可以通过 `fill_value` 参数切换为预填充模式：

- **在线推理服务**：需要确定性返回值，不能因缺失值而返回 None
- **已知缺失语义的业务**：例如"未填写的年龄默认为 0"
- **与上游特征填充策略对齐**：如果上游已经用 0 填充了缺失值

> ⚠️ **注意**：预填充可能改变运算语义。例如 `a/(a+b)` 中 b=None 填充为 0，结果变为 `a/(a+0)=1`，这在数学上可能不准确。请根据业务场景谨慎选择。

## 项目结构

```
RiskCraft/
├── feature_derivative/
│   ├── __init__.py          # 公共 API & transform() 统一入口
│   ├── exceptions.py        # 自定义异常类
│   ├── parser.py            # 表达式解析（变量提取 & AST 安全校验）
│   ├── sandbox.py           # 安全沙箱 eval()
│   ├── strategies.py        # 策略模式（Pandas / Spark / Online）
│   └── context.py           # 上下文类（自动引擎识别）
├── tests/
│   └── test_feature_derivative.py   # 测试套件
├── demo.py                  # 演示脚本
└── README.md
```

## 运行测试

```bash
# 基础测试（无需 PySpark）
pytest tests/test_feature_derivative.py -v

# 包含 Spark 测试（需安装 pyspark）
pip install pyspark
pytest tests/test_feature_derivative.py -v
```

## 运行演示

```bash
python demo.py
```
