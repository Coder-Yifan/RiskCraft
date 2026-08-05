# online_deploy_proto

protobuf 序列化的模型部署规格 + 自包含打分器 + PySpark 批量打分。

## 定位

`risk_ml/online_deploy` 用 **JSON** 序列化 `DeployPipeline`。JSON 有两个结构缺陷：

1. **类型还原 hack**：`json_safe` 把 dict 键 `str()` 化、±inf→`"Infinity"`、NaN→`None`，
   曾造成 max_diff 0.74 的静默误打分。
2. **无 schema / 版本演进**：JSON 无法表达字段编号与语义版本门控。

`online_deploy_proto` 新增一套 proto 序列化：numpy transformer 算子 + xgb 模型（onnx / m2cgen
双后端）+ proto **单产物**。同一份字节既可用于实时端打分，也可广播到 Spark 做批量打分。

## 架构

```
online_deploy_proto/            # import 轻量：只拖 numpy + protobuf（executor 安全）
├── deploy_spec.proto           # 版本演进契约
├── deploy_spec_pb2.py          # 已生成，入库（executor 免 protoc）
├── _kernels.py                 # executor 纯 numpy 5 算子内核（对照 risk_ml/_ops.py）
├── _model.py                   # onnx / m2cgen 打分引擎（onnxruntime 惰性加载）
├── scorer.py                   # ProtoScorer + build_engine（模块级缓存）+ RawOp 内核注册
├── codec.py                    # driver：DeployOp/模型 ↔ proto 直接转换（import risk_ml）
├── serialize.py                # driver：to_proto_bytes / from_proto_bytes
├── spark.py                    # Spark 3.2.4：mapInPandas + 固定 arity 标量 pandas_udf
└── codegen.py                  # 维护者：重新生成 deploy_spec_pb2.py
```

- **实时端 / Spark executor**：只 `import online_deploy_proto`（numpy + protobuf），
  绝不 import `risk_ml`（其 `__init__` 会拉 sklearn/xgboost/optuna 等重依赖）。
- **模型发布（driver）**：`serialize` / `codec` 才 import `risk_ml`。

## 快速开始

```python
# ---- 1. 发布（driver 侧）----
from risk_ml.online_deploy import PipelineParser
from online_deploy_proto.serialize import to_proto_bytes

deploy = PipelineParser(backend="m2cgen").compile_pipeline(pipe)
spec_bytes = to_proto_bytes(deploy)   # 单产物，可写入文件/DB/上传

# ---- 2. 实时打分（executor 侧，无 risk_ml）----
from online_deploy_proto import build_engine
scorer = build_engine(spec_bytes)
scorer.score({"amount": 3000, "age": 35})            # 单条
scorer.score_rows([{"amount": 3000}, {...}])         # 批量

# ---- 3. 还原为 DeployPipeline（driver 侧，可继续校验）----
from online_deploy_proto.serialize import from_proto_bytes
back = from_proto_bytes(spec_bytes)
back.score_batch(rows)
```

## PySpark 3.2.4 批量打分

```python
from pyspark.sql import SparkSession
from online_deploy_proto.spark import add_risk_score

spark = SparkSession.builder.getOrCreate()
df = spark.read.parquet("features/")          # 须含 spec 内 feature_names_in 全部列
scored = add_risk_score(df, spec_bytes)       # mapInPandas：原列 + risk_score
scored.write.mode("overwrite").parquet("scores/")
```

按列标量 UDF 方式：

```python
from pyspark.sql.functions import col
from online_deploy_proto.spark import make_scalar_pandas_udf

udf = make_scalar_pandas_udf(spec_bytes, deploy.feature_names_in_)
scored = df.withColumn("risk_score", udf(*[col(c) for c in deploy.feature_names_in_]))
```

集群提交（Spark 3.2.4；Python ≤3.10，Py3.12 不兼容）：

```bash
spark-submit \
  --py-files online_deploy_proto/ \
  --conf spark.sql.execution.arrow.pyspark.enabled=true \
  app.py
```

集群侧依赖：`numpy`、`protobuf>=6.31.1,<7`；onnx 后端另需 `onnxruntime`（m2cgen 后端零依赖）。

## 一致性保证

- **同产物 × 同解释器 × 同输入契约** → bit 级一致。`ProtoScorer` 内核与 `risk_ml/online_deploy/_ops.py`
  逐位对照复刻，测试在随机 / 分箱边界±eps / 缺失 / 哨兵样本上锁死 max_diff < 1e-9。
- **codec 直接构造，不走 `to_dict/from_dict`**：绕开 JSON 时代 `_num_key` 对原生 float 键的
  int() 截断 bug（`_num_key(2.5)=2`）。proto double 键原生保真，非整数值 float 分类键不再误伤。
- **单引擎跨端**：实时与 Spark 共用同一 proto 字节 + 同一 scorer 内核，天然一致。

## 与 JSON 对比

| | JSON (`online_deploy`) | proto (`online_deploy_proto`) |
|---|---|---|
| 类型还原 | `str` 键 / `"Infinity"` / NaN→None hack | 原生 double / bytes，零 hack |
| 版本演进 | 无 | 字段编号 + `min_scorer_version` 语义门控 |
| 体积 | onnx 需 base64 | 免 base64，onnx 后端约省 30% |
| 幂等/缓存 | 不保证 | proto3 map 排序 → 字节确定性 |
| 自定义算子 | `_OP_CLASSES` 写死 5 类，JSON 往返挂 | RawOp 逃生舱（driver codec + executor kernel 双注册） |

## 版本演进（字段编号）

proto 加字段**不破坏**旧版本读取（wire format 自动跳过未知字段），字段语义变更通过
`min_scorer_version` 显式门控——scorer 遇到未知 op 明确报错，绝不静默跳过（与 JSON 时代
静默误打分的教训相反）。字段编号约定：新增递增，废弃用 `reserved` 占位，绝不复用。

## 自定义算子（RawOp）

```python
# driver 侧：注册 编解码
from online_deploy_proto.codec import register_proto_op
register_proto_op("scale10", to_params_json, from_params_dict)

# executor 侧：注册 打分内核
from online_deploy_proto import register_scorer_kernel
register_scorer_kernel("scale10", builder)   # builder(op_params, input_idx) -> fn(X)->X_new
```

未注册时：driver 序列化 / executor 打分各自明确报错。

## 依赖与代码生成

- 运行时：`protobuf>=6.31.1,<7`（gencode major=6，与 protobuf 6.x 匹配）；`numpy`
- 可选：`onnx` / `onnxruntime`（onnx 后端）；`pyspark==3.2.4` + `pyarrow`（Spark）
- 仅维护者：`grpcio-tools==1.80.0`（捆绑 libprotoc 31.1 → gencode major=6）

重新生成 `deploy_spec_pb2.py`：`python online_deploy_proto/codegen.py`，随后验证
`python -c "from online_deploy_proto import deploy_spec_pb2 as pb; print(pb.DeploySpec.DESCRIPTOR.full_name)"`。

## 测试

```bash
pytest online_deploy_proto/tests/ -q --tb=short
```

- `test_codec.py`：round-trip 逐位一致 / ±inf·NaN 保真 / **cat float 键 2.5 回归** / 字节确定性 / RawOp
- `test_scorer_parity.py`：`ProtoScorer == DeployPipeline`（随机+边界+缺失+哨兵，<1e-9）/ RawOp 内核
- `test_spark_functions.py`：无 pyspark 可跑的纯函数（mapInPandas / exec 标量 UDF）
- `test_spark_integration.py`：需 pyspark（本机 skip）
