"""
risk_ml.online_deploy — 模型在线部署模块

将已拟合的 sklearn pipeline 编译为高性能、单条实时打分的部署流水线：
- transformer 算子用纯 numpy 复刻（分箱/WOE/清洗/筛选），与离线 predict 一致
- xgb 模型支持双后端（供用户选择）：
  - backend="m2cgen": 纯 Python 转译（float32 语义修正），~10us/条，零依赖
  - backend="onnx":   ONNX Runtime，~16us/条，跨语言可复用
- assert_consistent: 以 predict_proba 为真值，atol=1e-4 一致性校验
- JSON 序列化（无 pickle），支持自定义算子注册

Example
-------
>>> from risk_ml.online_deploy import PipelineParser, assert_consistent
>>> deploy = PipelineParser(backend="m2cgen").compile_pipeline(pipe)
>>> deploy.score({"amount": 3000, "age": 35})
>>> assert_consistent(pipe, deploy, X=X_test)
>>> deploy.to_json()   # 序列化，线上直接加载
"""

from ._base import DeployOp
from .benchmark import benchmark
from .checker import assert_consistent, generate_test_rows
from .exceptions import (
    ConsistencyError,
    DeployError,
    SerializationError,
    UnsupportedStepError,
)
from .parser import DeployPipeline, PipelineParser
from .registry import register_deploy_builder

__version__ = "0.1.0"

__all__ = [
    "DeployPipeline",
    "PipelineParser",
    "DeployOp",
    "assert_consistent",
    "generate_test_rows",
    "benchmark",
    "register_deploy_builder",
    "DeployError",
    "UnsupportedStepError",
    "ConsistencyError",
    "SerializationError",
]
