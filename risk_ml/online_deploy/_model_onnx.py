"""
ONNX 后端 — 从 xgboost booster 手工构建 ONNX TreeEnsemble

skl2onnx 已移除 xgboost 转换器、xgboost.to_onnx 已删除，
故从 booster 的 JSON dump 手工构建 ONNX 图（已验证与 predict_proba
偏差 ~1e-7，float32 舍入级别，0 行超 1e-4）。

依赖：onnx / onnxruntime（可选，仅选择 backend="onnx" 时需要）。
"""

import base64
import json
import math

import numpy as np

from ._base import json_safe


def logit(p):
    p = float(p)
    p = min(max(p, 1e-15), 1 - 1e-15)
    return math.log(p / (1.0 - p))


def build_onnx_model(booster, base_prob, feature_names, target_opset=17):
    """从 booster 构建 ONNX 图：float32[N,F] → TreeEnsemble(margin) → Sigmoid → prob。"""
    from onnx import helper, TensorProto

    feat_idx = {n: i for i, n in enumerate(feature_names)}
    trees = json.loads("[" + ",".join(booster.get_dump(dump_format="json")) + "]")

    treeids, nodeids, featids, modes, values = [], [], [], [], []
    truenodeids, falsenodeids, missing_true = [], [], []
    t_treeids, t_nodeids, t_weights = [], [], []

    for ti, tree in enumerate(trees):
        stack = [tree]
        while stack:
            node = stack.pop()
            nid = node["nodeid"]
            # 所有节点(内部+叶子)都要出现在 nodes_* 数组, 叶子用 LEAF mode
            treeids.append(ti)
            nodeids.append(nid)
            if "leaf" in node:
                featids.append(0)
                modes.append("LEAF")
                values.append(0.0)
                truenodeids.append(nid)
                falsenodeids.append(nid)
                missing_true.append(False)
                t_treeids.append(ti)
                t_nodeids.append(nid)
                t_weights.append(float(node["leaf"]))
                continue
            featids.append(feat_idx[node["split"]])
            modes.append("BRANCH_LT")  # value < threshold -> true(yes), 与 xgboost 一致
            values.append(float(node["split_condition"]))
            truenodeids.append(node["yes"])
            falsenodeids.append(node["no"])
            missing_true.append(node["missing"] == node["yes"])
            stack.extend(node["children"])

    tree_ensemble = helper.make_node(
        "TreeEnsembleRegressor",
        inputs=["input"],
        outputs=["margin"],
        domain="ai.onnx.ml",
        n_targets=1,
        nodes_treeids=treeids,
        nodes_nodeids=nodeids,
        nodes_featureids=featids,
        nodes_modes=modes,
        nodes_values=values,
        nodes_truenodeids=truenodeids,
        nodes_falsenodeids=falsenodeids,
        nodes_missing_value_tracks_true=missing_true,
        target_treeids=t_treeids,
        target_nodeids=t_nodeids,
        target_ids=[0] * len(t_weights),
        target_weights=t_weights,
        base_values=[logit(base_prob)],  # margin 偏移 = logit(概率)
        post_transform="NONE",
    )
    sigmoid = helper.make_node("Sigmoid", inputs=["margin"], outputs=["prob"])
    graph = helper.make_graph(
        [tree_ensemble, sigmoid],
        "xgb",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, len(feature_names)])],
        [helper.make_tensor_value_info("prob", TensorProto.FLOAT, [None, 1])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", target_opset), helper.make_opsetid("ai.onnx.ml", 3)],
    )
    model.ir_version = 8
    return model


class OnnxBackend:
    """ONNX Runtime 打分后端。

    编译期构建 ONNX blob（可 JSON 序列化为 base64），
    运行期懒加载 onnxruntime session，单条打分 ~20us。
    """

    def __init__(self, model_bytes, feature_names, base_score):
        self.model_bytes = bytes(model_bytes)
        self.feature_names = list(feature_names)
        self.base_score = float(base_score)
        self._session = None

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------
    @classmethod
    def from_booster(cls, booster, feature_names, base_score, opset=17):
        model = build_onnx_model(booster, base_score, feature_names, target_opset=opset)
        return cls(model.SerializeToString(), feature_names, base_score)

    # ------------------------------------------------------------------
    # 打分
    # ------------------------------------------------------------------
    def score(self, X):
        """批量打分，X 形状 (n, f) → 正例概率 (n,)。"""
        import onnxruntime as ort

        if self._session is None:
            self._session = ort.InferenceSession(
                self.model_bytes, providers=["CPUExecutionProvider"]
            )
        out = self._session.run(
            None, {"input": np.asarray(X, dtype=np.float32)}
        )[0]
        return out[:, 0]

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def to_dict(self):
        return {
            "kind": "onnx",
            "feature_names": self.feature_names,
            "base_score": self.base_score,
            "model_bytes_b64": base64.b64encode(self.model_bytes).decode("ascii"),
        }

    @classmethod
    def from_dict(cls, d):
        blob = base64.b64decode(d["model_bytes_b64"])
        return cls(blob, d["feature_names"], d["base_score"])

    def describe(self):
        return f"OnnxBackend({len(self.feature_names)} 特征, {len(self.model_bytes)} bytes)"
