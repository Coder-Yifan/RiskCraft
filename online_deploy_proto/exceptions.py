"""
online_deploy_proto 异常体系

与 risk_ml/online_deploy/exceptions.py 语义对齐，但独立于 risk_ml（executor 侧可用）。
"""


class DeploySpecError(Exception):
    """online_deploy_proto 异常基类。"""


class SerializationError(DeploySpecError):
    """proto 序列化 / 反序列化失败（版本不匹配、未知算子、字段缺失等）。"""


class ScoringError(DeploySpecError):
    """executor 打分链路失败（RawOp 内核未注册、引擎构建失败等）。"""
