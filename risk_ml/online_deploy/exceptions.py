"""
在线部署异常体系

统一捕获部署链路中的异常，便于调用方按类型处理。
"""


class DeployError(Exception):
    """在线部署模块的异常基类。"""


class UnsupportedStepError(DeployError):
    """pipeline 中存在无法编译的步骤。

    例如 FeatureCleaner 使用了缺失策略 drop_row（线上单条无法删行）、
    RiskXGBClassifier 训练时包含分类特征（ONNX/m2cgen 仅支持数值特征）。
    """


class ConsistencyError(DeployError):
    """一致性校验失败：部署打分与 sklearn pipeline 预测偏差超过 atol。"""


class SerializationError(DeployError):
    """部署流水线序列化 / 反序列化失败。"""
