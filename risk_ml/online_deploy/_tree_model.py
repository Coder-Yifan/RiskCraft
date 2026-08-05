"""
归一化树模型中间表示 — TreeModel

把各框架（xgboost / lightgbm）的 booster 归一化成一份框架无关的树结构，
让 m2cgen 代码生成 与 ONNX 图构建 共用同一套推理逻辑（统一线上推理）。

归一化时按框架量化 float32 语义：
- xgboost: 阈值/叶子 float32 舍入（f32），base_margin = f32(logit(base_prob))
- lightgbm: 阈值 float32 舍入，叶子保留 double，base_margin = 0
  （LGB 概率 = sigmoid(Σleaf)，实测 sigmoid(raw_score) == predict_proba，maxdiff 0.0）

归一化节点字段（dict）：
    is_leaf       : bool
    leaf          : float      # 叶子值（框架已量化）
    feature       : int        # 特征索引（非叶子）
    threshold     : float      # 阈值（框架已量化）
    mode          : "LT"|"LE"  # 比较模式：LT = value < threshold, LE = value <= threshold
    left          : dict       # 条件为真时的子节点（归一化）
    right         : dict       # 条件为假时的子节点（归一化）
    missing_left  : bool       # 缺失值是否走左（真）分支
"""

import json
import math
import struct


def f32(x):
    """float64 → float32 舍入后还原为 float64（值精确等于 float32）。"""
    return struct.unpack("f", struct.pack("f", float(x)))[0]


class TreeModel:
    """归一化树模型：feature_names + base_prob/base_margin + 树列表。"""

    __slots__ = ("feature_names", "base_prob", "base_margin", "trees")

    def __init__(self, feature_names, base_prob, base_margin, trees):
        self.feature_names = list(feature_names)
        self.base_prob = float(base_prob)
        self.base_margin = float(base_margin)
        self.trees = trees

    # ------------------------------------------------------------------
    # xgboost
    # ------------------------------------------------------------------
    @classmethod
    def from_xgb_booster(cls, booster, feature_names, base_score=0.5):
        """从 xgboost.Booster 归一化。

        base_score 为概率先验（来自 save_config）；阈值/叶子按 float32 语义舍入。
        """
        feat_idx = {n: i for i, n in enumerate(feature_names)}
        raw = json.loads("[" + ",".join(booster.get_dump(dump_format="json")) + "]")
        trees = [cls._norm_xgb_node(t, feat_idx) for t in raw]
        base_prob = float(base_score)
        base_margin = f32(math.log(base_prob / (1.0 - base_prob)))
        return cls(feature_names, base_prob, base_margin, trees)

    @classmethod
    def _norm_xgb_node(cls, node, feat_idx):
        if "leaf" in node:
            return {"is_leaf": True, "leaf": f32(node["leaf"])}
        return {
            "is_leaf": False,
            "feature": feat_idx[node["split"]],
            "threshold": f32(node["split_condition"]),
            "mode": "LT",  # xgboost 数值分裂: value < threshold 走 yes（左）
            "left": cls._norm_xgb_node(node["children"][0], feat_idx),
            "right": cls._norm_xgb_node(node["children"][1], feat_idx),
            "missing_left": node.get("missing", node["yes"]) == node["yes"],
        }

    # ------------------------------------------------------------------
    # lightgbm
    # ------------------------------------------------------------------
    @classmethod
    def from_lgb_booster(cls, booster, feature_names):
        """从 lightgbm.Booster 归一化。

        LGB 概率 = sigmoid(Σleaf)，无 base 偏移（base_margin = 0）；
        阈值按 float32 舍入，叶子保留 double 全精度。
        """
        dump = booster.dump_model()
        feat_idx = {n: i for i, n in enumerate(feature_names)}
        trees = [
            cls._norm_lgb_node(t["tree_structure"], feat_idx)
            for t in dump["tree_info"]
        ]
        return cls(feature_names, 0.5, 0.0, trees)

    @classmethod
    def _norm_lgb_node(cls, node, feat_idx):
        if "leaf_value" in node:
            return {"is_leaf": True, "leaf": float(node["leaf_value"])}
        f = node["split_feature"]
        fi = feat_idx[f] if isinstance(f, str) else int(f)
        dt = str(node.get("decision_type", "<=")).strip()
        if dt == "<=":
            mode = "LE"
        elif dt == "<":
            mode = "LT"
        else:
            raise NotImplementedError(
                f"LightGBM 非数值分裂 decision_type={dt!r} 暂不支持部署"
            )
        return {
            "is_leaf": False,
            "feature": fi,
            "threshold": f32(node["threshold"]),
            "mode": mode,
            "left": cls._norm_lgb_node(node["left_child"], feat_idx),
            "right": cls._norm_lgb_node(node["right_child"], feat_idx),
            "missing_left": bool(node.get("default_left", False)),
        }
