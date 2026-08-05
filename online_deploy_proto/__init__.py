"""
online_deploy_proto — proto 序列化 + 自包含打分器 + PySpark 批量打分

import 轻量：只拖 numpy + protobuf + deploy_spec_pb2（executor 安全，不触发 risk_ml）。
driver 侧功能（serialize / codec）依赖 risk_ml，spark 依赖 pyspark，均惰性导入。

用法
----
实时端打分（无 risk_ml，可进 Spark executor）：
    from online_deploy_proto import build_engine
    scorer = build_engine(spec_bytes)
    scorer.score({"amount": 3000, "age": 35})

模型发布（driver 侧）：
    from online_deploy_proto.serialize import to_proto_bytes, from_proto_bytes
    spec_bytes = to_proto_bytes(deploy)              # DeployPipeline → proto 字节
    deploy2 = from_proto_bytes(spec_bytes)           # 还原，可再打分/校验
"""

from .scorer import (
    ProtoScorer,
    __version__,
    build_engine,
    register_scorer_kernel,
)

__all__ = [
    "ProtoScorer",
    "build_engine",
    "register_scorer_kernel",
    "__version__",
]
