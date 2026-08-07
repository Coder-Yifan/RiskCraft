"""
m2cgen 后端 — 修正 float32 语义的纯 Python 转译

m2cgen 官方生成器用 float64 比较树阈值，而 xgboost 内部用 float32，
导致阈值边界行分支翻转（实测 max diff ~0.1）。本后端复刻其转译思路
并修正精度：阈值/叶子烘焙为 float32 舍入字面量，输入由包装器侧 numpy
向量化 float32 量化（astype float32→float64，替代逐元素 struct 循环），
纯 float64 比较精确等价于 float32 语义（实测 max diff ~1e-7，零翻转）。

性能优化（宽特征场景）：输入量化原先生成在代码里逐元素 struct 舍入，
成本 O(特征数)×(pack+unpack)。现移出生成代码，由 M2CgenBackend.score
用 numpy 一次向量化完成——生成代码仅做树遍历 + 累加，消除宽特征惩罚。

生成代码为纯标准库 Python（仅 math），零依赖，可内嵌字符串。
"""

import numpy as np

from ._base import json_safe
from ._tree_model import TreeModel, f32

# 兼容旧引用（如 risk_ml.tests.test_deploy 直接导入 _f32）
_f32 = f32


def transpile(tree_model, score_scaler=None):
    """将归一化 TreeModel 转译为 float32 修正版纯 Python score(input) 函数。

    评分拉伸折叠：logit(sigmoid(m)) ≡ m（m 为树模型 margin），故配置评分拉伸算子时
    末行直接 ``return offset + scale*raw``，省掉 sigmoid 的 exp 计算——线上打分只快不慢。

    Args:
        tree_model: _tree_model.TreeModel（阈值/叶子已按框架 float32 量化）
        score_scaler: ScoreScaler or None。None 时输出正例概率；配置后输出拉伸风险分。

    Returns:
        可直接 exec 的 Python 源码字符串
    """
    base_margin = tree_model.base_margin
    scale = offset = None
    if score_scaler is not None:
        # scale = -factor（分数越高风险越低，默认）或 +factor
        scale = -float(score_scaler.factor) if score_scaler.higher_is_safer else float(score_scaler.factor)
        offset = float(score_scaler.offset)

    lines = [
        "import math",
    ]
    if scale is None:
        lines += [
            "def sigmoid(x):",
            "    if x < 0.0:",
            "        z = math.exp(x)",
            "        return z / (1.0 + z)",
            "    return 1.0 / (1.0 + math.exp(-x))",
        ]
    lines += [
        "def score(input):",
        f"    # input: float32 舍入后的特征值 list（由 M2CgenBackend.score 向量化量化）",
        "",
    ]

    def gen_tree(node, out_var, indent):
        pad = "    " * indent
        if node["is_leaf"]:
            return [f"{pad}{out_var} = {node['leaf']!r}"]
        op = "<=" if node["mode"] == "LE" else "<"
        out = [f"{pad}if input[{node['feature']}] {op} {node['threshold']!r}:"]
        out += gen_tree(node["left"], out_var, indent + 1)
        out += [f"{pad}else:"]
        out += gen_tree(node["right"], out_var, indent + 1)
        return out

    var_names = []
    for ti, tree in enumerate(tree_model.trees):
        v = f"var{ti}"
        var_names.append(v)
        lines += gen_tree(tree, v, 1)
        lines += [""]

    lines += [
        f"    raw = {base_margin!r} + " + " + ".join(var_names),
    ]
    if scale is None:
        lines += ["    return sigmoid(raw)"]
    else:
        lines += [f"    return {offset!r} + {scale!r} * raw"]
    return "\n".join(lines)


class M2CgenBackend:
    """float32 修正版纯 Python 打分后端。

    编译期转译生成源码字符串（可 JSON 序列化），
    运行期惰性 exec，单条打分 ~10us，零第三方依赖。
    """

    def __init__(self, code, feature_names, base_score, score_meta=None):
        self.code = code
        self.feature_names = list(feature_names)
        self.base_score = float(base_score)
        # 评分拉伸元数据 {offset, factor, higher_is_safer} or None。
        # 注意不能命名 score——会遮蔽打分方法 score(X)。
        self.score_meta = score_meta
        self._fn = None

    @classmethod
    def from_booster(cls, booster, feature_names, base_score):
        tree_model = TreeModel.from_xgb_booster(booster, feature_names, base_score)
        return cls.from_tree_model(tree_model)

    @classmethod
    def from_tree_model(cls, tree_model, score_scaler=None):
        code = transpile(tree_model, score_scaler=score_scaler)
        score_meta = None
        if score_scaler is not None:
            score_meta = {
                "offset": float(score_scaler.offset),
                "factor": float(score_scaler.factor),
                "higher_is_safer": bool(score_scaler.higher_is_safer),
            }
        return cls(code, tree_model.feature_names, tree_model.base_prob, score_meta=score_meta)

    def score(self, X):
        """批量打分，X 形状 (n, f) → 正例概率 (n,)。

        输入 float32 量化在此向量化完成（astype float32→float64，
        与 struct._f32 舍入 bit 级一致），生成代码仅做树遍历。
        """
        if self._fn is None:
            ns = {}
            exec(compile(self.code, "<m2cgen>", "exec"), ns)
            self._fn = ns["score"]
        f = self._fn
        Xq = np.asarray(X, dtype=np.float64).astype(np.float32).astype(np.float64)
        return np.array([f(row.tolist()) for row in Xq])

    def to_dict(self):
        d = {
            "kind": "m2cgen",
            "feature_names": self.feature_names,
            "base_score": self.base_score,
            "code": self.code,
        }
        if self.score_meta is not None:
            d["score"] = self.score_meta
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(d["code"], d["feature_names"], d["base_score"], score_meta=d.get("score"))

    def describe(self):
        base = f"M2CgenBackend({len(self.feature_names)} 特征, {len(self.code.splitlines())} 行代码)"
        if self.score_meta is not None:
            s = self.score_meta
            sign = "-" if s["higher_is_safer"] else "+"
            base += f", score_scaler(score={s['offset']:.1f} {sign} {s['factor']:.2f}*logit)"
        return base
