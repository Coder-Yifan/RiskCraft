"""
在线部署算子基类 — DeployOp

每个部署算子封装一个 pipeline 步骤的 transform 逻辑，
全部使用纯 numpy 实现（不依赖 pandas / 原始算子类），
输入输出均为固定列序的 numpy 数组，保证单条打分的高性能与一致性。

列约定：
- input_columns / output_columns：字符串列名列表
- transform(X)：X 形状 (n, len(input_columns))，返回 (n, len(output_columns))
- transform_one(row)：单条 dict → dict，走 transform 同一内核

序列化约定：
- to_dict() 返回 JSON 友好 dict（无 numpy 类型，NaN 统一转 None）
- from_dict() 逆过程
"""

import json

import numpy as np

from .exceptions import SerializationError


def json_safe(obj):
    """递归将 numpy 标量 / NaN / ±inf 转换为 JSON 安全类型。

    规则：
    - NaN → None
    - +inf → "Infinity"，-inf → "-Infinity"（字符串，反序列化时 float() 还原）
    - numpy 标量 / 数组 → 原生类型 / list
    """
    if isinstance(obj, np.generic):
        return json_safe(obj.item())
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, float):
        if np.isnan(obj):
            return None
        if np.isposinf(obj):
            return "Infinity"
        if np.isneginf(obj):
            return "-Infinity"
        return obj
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def json_dumps(obj):
    """JSON 序列化（NaN 安全）。"""
    return json.dumps(json_safe(obj), ensure_ascii=False, allow_nan=False)


class DeployOp:
    """部署算子基类。"""

    # 子类覆盖：算子类型标识，用于 from_dict 分发
    kind = "base"

    def __init__(self, name, input_columns, output_columns):
        self.name = name
        self.input_columns = list(input_columns)
        self.output_columns = list(output_columns)
        self._input_idx = {c: i for i, c in enumerate(self.input_columns)}

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------
    def transform(self, X):
        """批量变换。

        Args:
            X: np.ndarray，形状 (n, len(input_columns))

        Returns:
            np.ndarray，形状 (n, len(output_columns))
        """
        raise NotImplementedError

    def transform_one(self, row):
        """单条 dict → dict（走 transform 同一内核）。"""
        arr = np.full(len(self.input_columns), np.nan)
        for i, c in enumerate(self.input_columns):
            v = row.get(c)
            if v is None:
                continue
            try:
                arr[i] = v
            except (TypeError, ValueError):
                pass  # 非数值值（如分类字符串）→ NaN，由后续 cat_map 重新映射
        out = self.transform(arr[None, :])[0]
        return {c: float(v) for c, v in zip(self.output_columns, out)}

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def to_dict(self):
        return {
            "kind": self.kind,
            "name": self.name,
            "input_columns": self.input_columns,
            "output_columns": self.output_columns,
        }

    @classmethod
    def from_dict(cls, d):
        op = cls(d["name"], d["input_columns"], d["output_columns"])
        return op

    # ------------------------------------------------------------------
    # 展示
    # ------------------------------------------------------------------
    def describe(self):
        return f"{self.__class__.__name__}({self.name}): {len(self.input_columns)}→{len(self.output_columns)} 列"

    def __repr__(self):
        return self.describe()


def _get_column(X, col):
    """按列名从 numpy 数组取列（内部用列索引缓存）。"""
    return X[:, self._input_idx[col]]
