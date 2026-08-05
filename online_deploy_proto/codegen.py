"""
维护者工具：重新生成 deploy_spec_pb2.py

需要 grpcio-tools（仅 codegen 时；生成物已入库，executor 无需 protoc）。
注意：捆绑 protoc 版本对应 gencode 的 major 版本，生成物要求相同 major 的
protobuf runtime（当前仓库声明 protobuf>=6.31.1,<7，即 gencode major=6）。
生成后务必立即验证导入：
    python -c "from online_deploy_proto import deploy_spec_pb2 as pb; print(pb.DeploySpec.DESCRIPTOR.full_name)"
"""

import os
import subprocess
import sys

_PROTO_DIR = os.path.dirname(os.path.abspath(__file__))
_PROTO = os.path.join(_PROTO_DIR, "deploy_spec.proto")
_OUT = os.path.join(_PROTO_DIR, "deploy_spec_pb2.py")


def regenerate():
    """调用 grpcio-tools 内置 protoc 重新生成 deploy_spec_pb2.py。"""
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            "-I",
            _PROTO_DIR,
            f"--python_out={_PROTO_DIR}",
            _PROTO,
        ]
    )
    print(f"已重新生成 {_OUT}")


if __name__ == "__main__":
    regenerate()
