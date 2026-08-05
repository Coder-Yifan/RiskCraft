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

import json
import math
import struct

import numpy as np

from ._base import json_safe


def _f32(x):
    """float64 → float32 舍入后还原为 float64（值精确等于 float32）。"""
    return struct.unpack("f", struct.pack("f", float(x)))[0]


def transpile(booster, base_prob, feature_names):
    """将 booster 转译为 float32 修正版纯 Python score(input) 函数。

    Args:
        booster: xgboost.Booster
        base_prob: 模型 base_score（概率）
        feature_names: 特征名列表（顺序与模型输入一致）

    Returns:
        可直接 exec 的 Python 源码字符串
    """
    feat_idx = {n: i for i, n in enumerate(feature_names)}
    trees = json.loads("[" + ",".join(booster.get_dump(dump_format="json")) + "]")
    base_margin = _f32(math.log(float(base_prob) / (1 - float(base_prob))))

    lines = [
        "import math",
        "def sigmoid(x):",
        "    if x < 0.0:",
        "        z = math.exp(x)",
        "        return z / (1.0 + z)",
        "    return 1.0 / (1.0 + math.exp(-x))",
        "def score(input):",
        f"    # input: float32 舍入后的特征值 list（由 M2CgenBackend.score 向量化量化）",
        "",
    ]

    def gen_tree(node, out_var, indent):
        pad = "    " * indent
        if "leaf" in node:
            return [f"{pad}{out_var} = {_f32(node['leaf'])!r}"]
        f = feat_idx[node["split"]]
        t = _f32(node["split_condition"])
        yes, no = node["children"]  # xgboost JSON: children[0] = yes 分支
        out = [f"{pad}if input[{f}] < {t!r}:"]
        out += gen_tree(yes, out_var, indent + 1)
        out += [f"{pad}else:"]
        out += gen_tree(no, out_var, indent + 1)
        return out

    var_names = []
    for ti, tree in enumerate(trees):
        v = f"var{ti}"
        var_names.append(v)
        lines += gen_tree(tree, v, 1)
        lines += [""]

    lines += [
        f"    raw = {base_margin!r} + " + " + ".join(var_names),
        "    return sigmoid(raw)",
    ]
    return "\n".join(lines)


class M2CgenBackend:
    """float32 修正版纯 Python 打分后端。

    编译期转译生成源码字符串（可 JSON 序列化），
    运行期惰性 exec，单条打分 ~10us，零第三方依赖。
    """

    def __init__(self, code, feature_names, base_score):
        self.code = code
        self.feature_names = list(feature_names)
        self.base_score = float(base_score)
        self._fn = None

    @classmethod
    def from_booster(cls, booster, feature_names, base_score):
        code = transpile(booster, base_score, feature_names)
        return cls(code, feature_names, base_score)

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
        return {
            "kind": "m2cgen",
            "feature_names": self.feature_names,
            "base_score": self.base_score,
            "code": self.code,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["code"], d["feature_names"], d["base_score"])

    def describe(self):
        return f"M2CgenBackend({len(self.feature_names)} 特征, {len(self.code.splitlines())} 行代码)"
