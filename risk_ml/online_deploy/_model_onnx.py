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
from ._tree_model import TreeModel


def logit(p):
    p = float(p)
    p = min(max(p, 1e-15), 1 - 1e-15)
    return math.log(p / (1.0 - p))


def build_onnx_model(tree_model, target_opset=17):
    """从归一化 TreeModel 构建 ONNX 图：float32[N,F] → TreeEnsemble(margin) → Sigmoid → prob。"""
    from onnx import helper, TensorProto

    feature_names = tree_model.feature_names
    base_prob = tree_model.base_prob

    treeids, nodeids, featids, modes, values = [], [], [], [], []
    truenodeids, falsenodeids, missing_true = [], [], []
    t_treeids, t_nodeids, t_weights = [], [], []

    for ti, tree in enumerate(tree_model.trees):
        # 两遍法：先前序收集节点（列表索引即节点 id），再统一填充数组，
        # 保证 nodes_* 各数组按同一顺序对齐（父节点先于子节点，id 前序）。
        order = []

        def collect(node):
            order.append(node)
            if not node["is_leaf"]:
                collect(node["left"])
                collect(node["right"])

        collect(tree)
        idx = {id(node): nid for nid, node in enumerate(order)}

        for nid, node in enumerate(order):
            # 所有节点(内部+叶子)都要出现在 nodes_* 数组, 叶子用 LEAF mode
            treeids.append(ti)
            nodeids.append(nid)
            if node["is_leaf"]:
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
            featids.append(node["feature"])
            modes.append("BRANCH_LEQ" if node["mode"] == "LE" else "BRANCH_LT")
            values.append(float(node["threshold"]))
            truenodeids.append(idx[id(node["left"])])
            falsenodeids.append(idx[id(node["right"])])
            missing_true.append(bool(node["missing_left"]))

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
        "tree_model",
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
        tree_model = TreeModel.from_xgb_booster(booster, feature_names, base_score)
        return cls.from_tree_model(tree_model, opset=opset)

    @classmethod
    def from_tree_model(cls, tree_model, opset=17):
        model = build_onnx_model(tree_model, target_opset=opset)
        return cls(model.SerializeToString(), tree_model.feature_names, tree_model.base_prob)

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
