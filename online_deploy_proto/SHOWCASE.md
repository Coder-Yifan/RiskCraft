# 线上 / 线下，同一份文件打分

> 一次训练，产出一份自包含的 `.pbin` 部署文件。离线校验、在线实时、Spark 批量，
> 全部消费**同一份字节**——线上线下打分**零分叉、零重写**。

---

## 一句话卖点

**模型 + 全链路预处理算子 + 特征清单 + 版本契约，打包成一份 proto 字节文件。**
这一份文件同时覆盖**线上实时**与**线下批量**两种打分场景：

- **线上·在线实时**（executor）：解析成自包含 `ProtoScorer`，单条实时打分，**不依赖 risk_ml**；
- **线下·离线批量**，两条路任选：
  1. **pkl 打分** — sklearn pipeline 离线参考/校验（真值基准）；
  2. **Spark 批量打分** — 同一份 proto 文件，大数据批量打分。

```
     训练               编译                序列化                       双端消费
 ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────────────────────┐
 │ sklearn  │    │ PipelineParser│    │ to_proto_bytes│   │ 线上·实时: build_engine     │
 │ RiskPipeline│ → │  (cleaner→woe │ →  │ ────────────│→  │   → ProtoScorer.score(row) │
 │ .fit(X,y) │    │  →select→模型) │    │  model.pbin │    │ 线下·批量:                 │
 └──────────┘    └──────────────┘    └──────────────┘    │   ① pkl 打分（参考/校验）    │
                                                          │   ② Spark 批量（同一份proto）│
                                                          └────────────────────────────┘
                                                           一份文件，覆盖线上实时 + 线下批量
```

---

## 模型发布（driver 侧，一次产出）

```python
from risk_ml import RiskPipeline
from risk_ml.encoding import BinnerWoeEncoder
from risk_ml.estimator import RiskXGBClassifier
from risk_ml.feature_selection import IVSelector
from risk_ml.online_deploy import PipelineParser
from risk_ml.preprocessing import FeatureCleaner
from online_deploy_proto.serialize import to_proto_bytes

pipe = RiskPipeline([
    ("cleaner", FeatureCleaner(sentinels=[-999])),
    ("bin_woe", BinnerWoeEncoder(max_bins=6)),
    ("select", IVSelector(iv_threshold=0.02)),
    ("model", RiskXGBClassifier(n_estimators=100, max_depth=4)),
]).fit(X_train, y_train)

# 编译成部署流水线，选 onnx（跨语言）或 m2cgen（零依赖）
deploy = PipelineParser(backend="onnx").compile_pipeline(pipe)

# 产出单份 proto 字节 —— 这就是要发布/下发的那个文件
spec_bytes = to_proto_bytes(deploy)
with open("model.pbin", "wb") as f:
    f.write(spec_bytes)

# 可选：另存 pkl，供线下参考/校验打分（真值基准）
import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(pipe, f)
```

发布物是 `model.pbin` —— 一份自包含，同时供给线上实时和 Spark 线下批量两条链路。
`model.pkl` 只是可选的线下参考副本（真值基准），不参与线上链路。

---

## 线下打分（离线批量）

线下打分两条路：pkl（sklearn 参考）与 Spark 批量（同一份 proto）。

### ① pkl 打分（sklearn 参考 / 校验）

```python
import pickle

with open("model.pkl", "rb") as f:
    pipe = pickle.load(f)                  # pickle 保存的 sklearn pipeline

scores = pipe.predict_proba(X_test)[:, 1]  # sklearn 全量离线打分（真值基准）
```

### ② Spark 批量打分（同一份 proto 文件，大数据批量）

```python
from pyspark.sql import SparkSession
from online_deploy_proto.spark import add_risk_score

spark = SparkSession.builder.getOrCreate()
df = spark.read.parquet("features/")      # 须含 spec 内 feature_names_in 全部列
scored = add_risk_score(df, spec_bytes)   # mapInPandas：原列 + risk_score
scored.write.mode("overwrite").parquet("scores/")
```

`add_risk_score` 消费的就是线上那份 `model.pbin` 的字节——Spark executor 内
`build_engine` 解析出同一套内核，因此与线上打分、pkl 参考天然一致。

> 发布自检（可选）：`from_proto_bytes(spec_bytes)` 可把同一份字节还原成
> `DeployPipeline` 做往返校验，但这**不是**线下打分的日常路径。

---

## 线上打分（在线实时，零 risk_ml 依赖）

```python
from online_deploy_proto import build_engine

with open("model.pbin", "rb") as f:
    spec_bytes = f.read()

scorer = build_engine(spec_bytes)          # 解析 + 模块级缓存（按字节去重）
scorer.score({"amount": 3000, "age": 35})  # 单条实时打分 → 正例概率
```

三种入口，覆盖实时 / 批量 / 数组：

```python
scorer.score(row_dict)                     # 单条：dict → float
scorer.score_rows([{...}, {...}])          # 批量：list[dict] → np.ndarray(n,)
scorer.score_np(np_array)                  # 批量：array (n, f) → np.ndarray(n,)
```

executor 侧 **绝不 import risk_ml**——`import online_deploy_proto` 只拖
numpy + protobuf（onnxruntime / pyspark 均为惰性导入），可以放心进入
API 服务进程或 Spark executor。

---

## 为什么「一份文件」就够：三个自包含

| 维度 | 说明 |
|------|------|
| **全链路算子** | cleaner / bin / woe / select / derive 等所有预处理算子，以参数化 proto 消息内嵌，executor 用纯 numpy 内核逐位复刻线下算子（derive = feature_derivative 表达式转译源码，线下线上同一份代码） |
| **模型后端** | onnx 后端内嵌标准 ONNX 二进制（`model_bytes`，免 base64）；m2cgen 后端内嵌转译 Python 源码（`code` 字符串），都是自包含可执行 |
| **schema 版本契约** | `DeploySpec` 带 `version` + `min_scorer_version`（语义化），scorer 版本过旧加载新文件时**明确报错**而非静默打分错误 |

配套保证：

- **字节确定性**：proto3 map 按 key 排序，同一 deploy 两次 `to_proto_bytes` 字节完全一致 → 可直接做缓存 key / 完整性校验。
- **自定义算子逃生舱**：不落入六类内置算子的算子，用 `RawOp`（params_json）+ 双端注册表扩展——
  driver 侧 `register_proto_op`、executor 侧 `register_scorer_kernel`，同一 kind 对应同一内核。
- **跨语言**：onnx 后端是标准 ONNX 模型，任何支持 ONNX Runtime 的语言（Java / Go / C++…）都能直接消费同一份 `model.pbin`。

---

## 一致性：以 pkl（sklearn 离线）为准

线上 `ProtoScorer`、线下 Spark 批量都消费**同一份字节**、复用同一套打分内核，
与 pkl（sklearn）全量离线结果逐位对齐。锁定方式：

```python
import numpy as np

truth = pipe.predict_proba(X_test)[:, 1]                 # 离线 sklearn 全量
online = scorer.score_rows(X_test.to_dict("records"))    # 线上同一份字节

print(np.max(np.abs(online - truth)))                    # ~1e-7
print(np.sum(np.abs(online - truth) > 1e-4))             # 0
```

一致性由测试套件锁死：`online_deploy_proto/tests/test_scorer_parity.py` 断言
`ProtoScorer` 与 `DeployPipeline` 在随机 + 边界 ±eps + 缺失 + 哨兵样本上逐位一致
（max_diff < 1e-9）。

实测（`demo_deploy_compare.py`，双 pipeline，500 行抽样，atol=1e-4，以 pkl 为准）：

| case | proto+m2cgen | proto+onnx |
|---|---|---|
| xgb | max_diff 8.6e-08，0/500 超阈值 | max_diff 1.8e-07，0/500 超阈值 |
| lgb | max_diff 1.1e-16（bit-exact），0/500 | max_diff 8.3e-08，0/500 |

---

## 性能：线上单条 35~65x vs pkl

线上单条实时打分（`demo_deploy_compare.py` 实测，6 特征 / 100 树 / 2 特征入选）：

| case | pkl | proto+m2cgen | proto+onnx |
|---|---|---|---|
| xgb | 7332 us | **136 us（53.9x）** | 209 us（35.0x） |
| lgb | 9012 us | **138 us（65.4x）** | 193 us（46.7x） |

- 后端选择：窄特征实时场景 m2cgen 略优（纯 Python，无 session 往返）；需要**跨语言**或未来上别的推理框架时选 onnx。
- pkl 慢的绝对大头是 sklearn pipeline 在单行 DataFrame 上的 transform 开销，这正是部署文件规避的部分。
- 线下批量：大批量离线打分走 Spark（同一份 proto），不经过 sklearn / pkl 的单行 DataFrame 开销。

发布物大小对比：

| case | proto+m2cgen | proto+onnx | pkl（仅参考） |
|---|---|---|---|
| xgb | 41 KB | 26 KB | 99 KB |
| lgb | 120 KB | 67 KB | 140 KB |

proto 是单产物自包含，比 pkl 小一个量级，且不含任何环境绑定。

---

## 完整闭环 demo

`risk_ml/online_deploy/demo_deploy_compare.py`（xgb + lgb 双案例）：

```bash
PYTHONIOENCODING=utf-8 PYTHONPATH="D:/project/RiskCraft" \
  D:/softwares/conda/python.exe risk_ml/online_deploy/demo_deploy_compare.py
```

每案例完成：训练 → 编译双后端 → 产出 `proto+onnx / proto+m2cgen / pkl` 三个文件 →
以 pkl 为准校验一致性 → 单条打分性能对比。运行后部署文件落在
`%TEMP%\riskcraft_deploy_compare\`。

---

## 快速参考

| 场景 | 入口 | 依赖 |
|---|---|---|
| 训练 + 编译 + 序列化 | `PipelineParser(...).compile_pipeline(pipe)` → `to_proto_bytes(deploy)` | risk_ml（driver） |
| 线上·实时单条 | `build_engine(spec_bytes).score(row)` | 仅 numpy + protobuf |
| 线上·批量 | `build_engine(spec_bytes).score_rows(rows)` / `.score_np(X)` | 仅 numpy + protobuf |
| 线下·pkl 打分 | `pickle.load(...)` → `predict_proba(X)` | risk_ml（sklearn，参考/校验） |
| 线下·Spark 批量 | `online_deploy_proto.spark.add_risk_score(df, spec_bytes)` | pyspark（惰性） |
| 发布自检（往返还原） | `from_proto_bytes(spec_bytes).score_batch(rows)` | risk_ml（driver） |
| 自定义算子 | `register_proto_op`（driver）/ `register_scorer_kernel`（executor） | 双端各注册 |
