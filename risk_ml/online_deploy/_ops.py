"""
纯 numpy transformer 算子

用 numpy 复刻 risk_ml 各 transformer 的 transform 逻辑（pd.cut ↔ np.searchsorted、
fillna ↔ np.where、map ↔ 查表），保证单条打分与 sklearn pipeline 完全一致。

支持的 pipeline 步骤：
- FeatureCleaner      → CleanerOp
- ChiMergeBinner      → BinOp
- WoeEncoder          → WoeOp
- BinnerWoeEncoder    → BinWoeOp
- IVSelector / CorrelationSelector → SelectOp

注意：部署仅支持数值特征 pipeline；字符串分类特征需预先编码为数值。
"""

import numpy as np

from ._base import DeployOp
from .exceptions import UnsupportedStepError


# ======================================================================
# 工具：pd.cut 等价实现
# ======================================================================
def np_cut_labels(x, edges):
    """等价 pd.cut(x, bins=edges, labels=False, include_lowest=True)。

    返回整数箱索引 0..k-1；NaN 或越界返回 -1（表示缺失箱）。

    pandas 语义：right=True 时区间为 [e0,e1], (e1,e2], ..., (e_{k-1},e_k]，
    即 x 落在 edges[i]~edges[i+1] 的 i 箱；x 越界或为 NaN → 缺失箱。
    """
    x = np.asarray(x, dtype=np.float64)
    e = np.asarray(edges, dtype=np.float64)
    n_bins = len(e) - 1
    if n_bins <= 0:
        return np.full(x.shape, -1, dtype=np.int64)
    # side='left' 返回第一个 e[j] >= x 的 j，idx = j-1 表示 x 所在箱的下界索引
    idx = np.searchsorted(e, x, side="left") - 1
    out = np.full(x.shape, -1, dtype=np.int64)
    valid = (idx >= 0) & (idx <= n_bins - 1)
    out[valid] = idx[valid]
    # x == e[0]：searchsorted 返回 0 → idx=-1，但应归第 0 箱
    out[x == e[0]] = 0
    # NaN 显式置缺失箱（searchsorted 对 NaN 返回末尾，已越界，保险起见再置）
    out[np.isnan(x)] = -1
    return out


def _is_numeric_mappable(cat_map):
    """判断分类映射的键是否全部为数值（可安全塞入 float 数组）。"""
    return all(isinstance(k, (int, float)) for k in cat_map)


def _num(v):
    """JSON 还原：'Infinity'/'-Infinity' → ±inf，其余转 float。"""
    if isinstance(v, str) and v in ("Infinity", "-Infinity"):
        return float(v)
    return float(v)


def _num_key(k):
    """JSON 还原 dict 键：优先 int，其次 float，兜底保留原样。"""
    try:
        return int(k)
    except (ValueError, TypeError):
        try:
            return float(k)
        except (ValueError, TypeError):
            return k


# ======================================================================
# CleanerOp — FeatureCleaner
# ======================================================================
class CleanerOp(DeployOp):
    """特征清洗：哨兵→NaN → 删列 → 填充缺失 → 异常值截断/置NaN。"""

    kind = "cleaner"

    def __init__(self, name, input_columns, output_columns, sentinels,
                 impute_values, clip_bounds, outlier_action):
        super().__init__(name, input_columns, output_columns)
        self.sentinels = list(sentinels)
        self.impute_values = impute_values  # {col: fill_val}
        self.clip_bounds = clip_bounds      # {col: (lo, hi)}
        self.outlier_action = outlier_action

    @classmethod
    def from_step(cls, step, input_columns, name=""):
        if step.missing_strategy == "drop_row":
            raise UnsupportedStepError(
                f"[{name or 'cleaner'}] missing_strategy='drop_row' 无法在线部署"
                "（线上单条场景不能删除数据行），请改用 median/mean/constant 填充"
            )
        drop = set(step.drop_columns_)
        output = [c for c in input_columns if c not in drop]
        sentinels = step.sentinels if step.sentinels is not None else [-999, -9998, -9996]
        return cls(
            name or "cleaner", input_columns, output,
            sentinels=sentinels,
            impute_values=dict(step.impute_values_),
            clip_bounds=dict(step.clip_bounds_),
            outlier_action=step.outlier_action,
        )

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64).copy()
        n = X.shape[0]
        # 1. 哨兵值 → NaN（部署要求数值特征，逐列替换语义与 pandas replace 一致）
        if self.sentinels:
            for i in range(X.shape[1]):
                col = X[:, i]
                for s in self.sentinels:
                    col[col == s] = np.nan
        # 2. 删列 + 填充 + 截断，按 output_columns 顺序组装
        out = np.full((n, len(self.output_columns)), np.nan)
        for j, col in enumerate(self.output_columns):
            col_data = X[:, self._input_idx[col]].copy()
            fill = self.impute_values.get(col)
            if fill is not None:
                col_data = np.where(np.isnan(col_data), fill, col_data)
            if col in self.clip_bounds:
                lo, hi = self.clip_bounds[col]
                if self.outlier_action == "clip":
                    col_data = np.clip(col_data, lo, hi)
                else:  # set_nan
                    col_data = np.where((col_data < lo) | (col_data > hi), np.nan, col_data)
            out[:, j] = col_data
        return out

    def to_dict(self):
        d = super().to_dict()
        d.update({
            "sentinels": self.sentinels,
            "impute_values": self.impute_values,
            "clip_bounds": {k: list(v) for k, v in self.clip_bounds.items()},
            "outlier_action": self.outlier_action,
        })
        return d

    @classmethod
    def from_dict(cls, d):
        op = cls(
            d["name"], d["input_columns"], d["output_columns"],
            sentinels=d["sentinels"],
            impute_values=d["impute_values"],
            clip_bounds={k: (_num(v[0]), _num(v[1])) for k, v in d["clip_bounds"].items()},
            outlier_action=d["outlier_action"],
        )
        return op


# ======================================================================
# BinOp — BaseBinner（ChiMergeBinner）
# ======================================================================
class BinOp(DeployOp):
    """分箱：连续列 pd.cut 等价，分类列先 cat_map 再 cut。"""

    kind = "bin"

    def __init__(self, name, input_columns, output_columns, edges, cat_maps=None):
        super().__init__(name, input_columns, output_columns)
        self.edges = edges        # {col: [边界]}
        self.cat_maps = cat_maps or {}  # {col: {orig: 整数编码}}
        self.categorical = set(self.cat_maps)

    @classmethod
    def from_step(cls, step, input_columns, name=""):
        edges = {c: np.asarray(v, dtype=np.float64).tolist()
                 for c, v in step.bin_edges_.items()}
        cat_maps = {}
        if hasattr(step, "_categorical_cols_"):
            for col in step._categorical_cols_:
                if col not in input_columns:
                    continue
                cmap = step._cat_maps_.get(col, {})
                if not _is_numeric_mappable(cmap):
                    raise UnsupportedStepError(
                        f"[{name or 'bin'}] 分类列 '{col}' 的类别值非数值"
                        "（如字符串），在线部署要求先对分类特征做数值编码"
                    )
                cat_maps[col] = dict(cmap)
        return cls(name or "bin", input_columns, input_columns, edges, cat_maps)

    def _bin_col(self, col, x):
        """单列分箱：连续列直接 cut；分类列先映射整数再 cut。"""
        if col in self.categorical:
            cmap = self.cat_maps[col]
            mapped = np.full(x.shape, np.nan)
            for orig, code in cmap.items():
                mapped[x == orig] = code
            return np_cut_labels(mapped, self.edges[col])
        return np_cut_labels(x, self.edges[col])

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        out = np.full((n, len(self.output_columns)), np.nan)
        for j, col in enumerate(self.output_columns):
            out[:, j] = self._bin_col(col, X[:, self._input_idx[col]])
        return out

    def to_dict(self):
        d = super().to_dict()
        d.update({"edges": self.edges, "cat_maps": self.cat_maps})
        return d

    @classmethod
    def from_dict(cls, d):
        edges = {c: [_num(x) for x in e] for c, e in d["edges"].items()}
        cat_maps = {c: {_num_key(k): v for k, v in m.items()}
                    for c, m in (d.get("cat_maps") or {}).items()}
        return cls(d["name"], d["input_columns"], d["output_columns"], edges, cat_maps)


# ======================================================================
# WoeOp — WoeEncoder
# ======================================================================
class WoeOp(DeployOp):
    """WOE 编码：箱索引 → WOE 值，未命中（含缺失箱）→ NaN。"""

    kind = "woe"

    def __init__(self, name, input_columns, output_columns, woe_maps):
        super().__init__(name, input_columns, output_columns)
        self.woe_maps = woe_maps  # {col: {bin_idx: woe}}

    @classmethod
    def from_step(cls, step, input_columns, name=""):
        woe_maps = {c: {int(k): float(v) for k, v in m.items()}
                    for c, m in step.woe_map_.items()}
        return cls(name or "woe", input_columns, input_columns, woe_maps)

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        out = np.full((n, len(self.output_columns)), np.nan)
        for j, col in enumerate(self.output_columns):
            mapped = np.full(n, np.nan)
            xj = X[:, self._input_idx[col]]
            for bin_idx, woe in self.woe_maps[col].items():
                mapped[xj == bin_idx] = woe
            out[:, j] = mapped
        return out

    def to_dict(self):
        d = super().to_dict()
        d.update({"woe_maps": self.woe_maps})
        return d

    @classmethod
    def from_dict(cls, d):
        woe_maps = {c: {int(k): float(v) for k, v in m.items()}
                    for c, m in d["woe_maps"].items()}
        return cls(d["name"], d["input_columns"], d["output_columns"], woe_maps)


# ======================================================================
# BinWoeOp — BinnerWoeEncoder
# ======================================================================
class BinWoeOp(DeployOp):
    """分箱 + WOE 一步完成（对应 BinnerWoeEncoder）。"""

    kind = "bin_woe"

    def __init__(self, name, input_columns, output_columns, edges, woe_maps, cat_maps=None):
        super().__init__(name, input_columns, output_columns)
        self.edges = edges
        self.woe_maps = woe_maps
        self.cat_maps = cat_maps or {}
        self.categorical = set(self.cat_maps)

    @classmethod
    def from_step(cls, step, input_columns, name=""):
        edges = {c: np.asarray(v, dtype=np.float64).tolist()
                 for c, v in step.bin_edges_.items()}
        woe_maps = {c: {int(k): float(v) for k, v in m.items()}
                    for c, m in step.woe_map_.items()}
        cat_maps = {}
        if hasattr(step.binner_, "_categorical_cols_"):
            for col in step.binner_._categorical_cols_:
                if col not in input_columns:
                    continue
                cmap = step.binner_._cat_maps_.get(col, {})
                if not _is_numeric_mappable(cmap):
                    raise UnsupportedStepError(
                        f"[{name or 'bin_woe'}] 分类列 '{col}' 的类别值非数值"
                        "（如字符串），在线部署要求先对分类特征做数值编码"
                    )
                cat_maps[col] = dict(cmap)
        return cls(name or "bin_woe", input_columns, input_columns, edges, woe_maps, cat_maps)

    def _bin_col(self, col, x):
        if col in self.categorical:
            cmap = self.cat_maps[col]
            mapped = np.full(x.shape, np.nan)
            for orig, code in cmap.items():
                mapped[x == orig] = code
            return np_cut_labels(mapped, self.edges[col])
        return np_cut_labels(x, self.edges[col])

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        # 先分箱
        binned = np.full((n, len(self.output_columns)), np.nan)
        for j, col in enumerate(self.output_columns):
            binned[:, j] = self._bin_col(col, X[:, self._input_idx[col]])
        # 再 WOE
        out = np.full((n, len(self.output_columns)), np.nan)
        for j, col in enumerate(self.output_columns):
            for bin_idx, woe in self.woe_maps[col].items():
                out[binned[:, j] == bin_idx, j] = woe
        return out

    def to_dict(self):
        d = super().to_dict()
        d.update({"edges": self.edges, "woe_maps": self.woe_maps, "cat_maps": self.cat_maps})
        return d

    @classmethod
    def from_dict(cls, d):
        edges = {c: [_num(x) for x in e] for c, e in d["edges"].items()}
        woe_maps = {c: {int(k): float(v) for k, v in m.items()}
                    for c, m in d["woe_maps"].items()}
        cat_maps = {c: {_num_key(k): v for k, v in m.items()}
                    for c, m in (d.get("cat_maps") or {}).items()}
        return cls(d["name"], d["input_columns"], d["output_columns"],
                   edges, woe_maps, cat_maps)


# ======================================================================
# SelectOp — RiskSelector（IVSelector / CorrelationSelector）
# ======================================================================
class SelectOp(DeployOp):
    """特征筛选：按保留列取子集。"""

    kind = "select"

    def __init__(self, name, input_columns, output_columns):
        super().__init__(name, input_columns, output_columns)

    @classmethod
    def from_step(cls, step, input_columns, name=""):
        mask = step._get_support_mask()
        keep = [c for c, m in zip(step.feature_names_in_, mask) if m]
        return cls(name or "select", input_columns, keep)

    def transform(self, X):
        idx = [self._input_idx[c] for c in self.output_columns]
        return np.asarray(X, dtype=np.float64)[:, idx]
