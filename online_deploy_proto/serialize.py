"""
driver 侧序列化门面：DeployPipeline ↔ proto 字节

to_proto_bytes 产出单产物字节（幂等、确定性：proto3 map 按 key 排序，字节可作缓存 key）。
from_proto_bytes 经 codec 直接构造 DeployPipeline（version / min_scorer_version 双重门控）。
"""

from . import deploy_spec_pb2 as pb
from .codec import model_to_proto, op_to_proto, proto_to_model, proto_to_op
from .exceptions import SerializationError
from .scorer import __version__, check_min_scorer_version


def to_proto_bytes(deploy):
    """DeployPipeline → proto 字节（含 min_scorer_version=当前包版本）。"""
    spec = pb.DeploySpec()
    spec.version = 1
    spec.feature_names_in.extend(deploy.feature_names_in_)
    for op in deploy.ops:
        spec.ops.append(op_to_proto(op))
    spec.model.CopyFrom(model_to_proto(deploy.model_op))
    spec.min_scorer_version = __version__
    return spec.SerializeToString()


def from_proto_bytes(data):
    """proto 字节 → DeployPipeline（driver 侧，可继续用 score/score_batch）。"""
    spec = pb.DeploySpec()
    spec.ParseFromString(data)
    if spec.version != 1:
        raise SerializationError(
            f"不支持的部署格式版本: {spec.version}（当前仅支持 1）"
        )
    check_min_scorer_version(spec.min_scorer_version)
    ops = [proto_to_op(m) for m in spec.ops]
    model = proto_to_model(spec.model)
    from risk_ml.online_deploy.parser import DeployPipeline

    return DeployPipeline(ops, model, list(spec.feature_names_in))
