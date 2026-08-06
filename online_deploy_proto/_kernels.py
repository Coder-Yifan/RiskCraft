"""
executor 侧纯 numpy 算子内核（禁止 import risk_ml）

与 risk_ml/online_deploy/_ops.py 逐位一致（对照抄写，禁止漂移）。
任何 _ops.py 的 transform 逻辑改动必须同步本文件，并由
tests/test_scorer_parity.py 以紧 atol 锁死两侧一致性。

本模块只依赖 numpy，供 Spark executor 反序列化后独立打分，
不触发 risk_ml 顶层 __init__ 的重依赖导入。

内核签名统一：
    kernel_xxx(X, input_idx, output_columns, ...) -> X_new
- X: (n, len(input_columns)) float64
- input_idx: {列名: 列位置}，按上游数组当前列序构造
- output_columns: 输出列名列表，决定返回数组的列
"""

import numpy as np

# ======================================================================
# 工具：pd.cut 等价（对照 _ops.py:25-47）
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


# ======================================================================
# CleanerOp 内核（对照 _ops.py:103-126）
# ======================================================================
def kernel_cleaner(X, input_idx, output_columns, sentinels,
                   impute_values, clip_bounds, outlier_action):
    """特征清洗：哨兵→NaN → 删列 → 填充缺失 → 异常值截断/置NaN。"""
    X = np.asarray(X, dtype=np.float64).copy()
    n = X.shape[0]
    # 1. 哨兵值 → NaN（部署要求数值特征，逐列替换语义与 pandas replace 一致）
    if sentinels:
        for i in range(X.shape[1]):
            col = X[:, i]
            for s in sentinels:
                col[col == s] = np.nan
    # 2. 删列 + 填充 + 截断，按 output_columns 顺序组装
    out = np.full((n, len(output_columns)), np.nan)
    for j, col in enumerate(output_columns):
        col_data = X[:, input_idx[col]].copy()
        fill = impute_values.get(col)
        if fill is not None:
            col_data = np.where(np.isnan(col_data), fill, col_data)
        if col in clip_bounds:
            lo, hi = clip_bounds[col]
            if outlier_action == "clip":
                col_data = np.clip(col_data, lo, hi)
            else:  # set_nan
                col_data = np.where((col_data < lo) | (col_data > hi), np.nan, col_data)
        out[:, j] = col_data
    return out


# ======================================================================
# BinOp 内核（对照 _ops.py:191-197, _bin_col:181-189）
# ======================================================================
def kernel_bin(X, input_idx, output_columns, edges, cat_maps):
    """分箱：连续列 pd.cut 等价，分类列先 cat_map 再 cut。"""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    categorical = set(cat_maps)
    out = np.full((n, len(output_columns)), np.nan)
    for j, col in enumerate(output_columns):
        x = X[:, input_idx[col]]
        if col in categorical:
            cmap = cat_maps[col]
            mapped = np.full(x.shape, np.nan)
            for orig, code in cmap.items():
                mapped[x == orig] = code
            out[:, j] = np_cut_labels(mapped, edges[col])
        else:
            out[:, j] = np_cut_labels(x, edges[col])
    return out


# ======================================================================
# WoeOp 内核（对照 _ops.py:229-246）
# ======================================================================
def kernel_woe(X, input_idx, output_columns, woe_maps):
    """WOE 编码：箱索引 → WOE 值，未命中（含缺失箱）→ NaN。"""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    out = np.full((n, len(output_columns)), np.nan)
    for j, col in enumerate(output_columns):
        mapped = np.full(n, np.nan)
        xj = X[:, input_idx[col]]
        for bin_idx, woe in woe_maps[col].items():
            mapped[xj == bin_idx] = woe
        out[:, j] = mapped
    return out


# ======================================================================
# BinWoeOp 内核（对照 _ops.py:304-316）
# ======================================================================
def kernel_bin_woe(X, input_idx, output_columns, edges, woe_maps, cat_maps):
    """分箱 + WOE 一步完成（对应 BinnerWoeEncoder）。"""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    categorical = set(cat_maps)
    # 先分箱
    binned = np.full((n, len(output_columns)), np.nan)
    for j, col in enumerate(output_columns):
        x = X[:, input_idx[col]]
        if col in categorical:
            cmap = cat_maps[col]
            mapped = np.full(x.shape, np.nan)
            for orig, code in cmap.items():
                mapped[x == orig] = code
            binned[:, j] = np_cut_labels(mapped, edges[col])
        else:
            binned[:, j] = np_cut_labels(x, edges[col])
    # 再 WOE
    out = np.full((n, len(output_columns)), np.nan)
    for j, col in enumerate(output_columns):
        for bin_idx, woe in woe_maps[col].items():
            out[binned[:, j] == bin_idx, j] = woe
    return out


# ======================================================================
# SelectOp 内核（对照 _ops.py:351-353）
# ======================================================================
def kernel_select(X, input_idx, output_columns):
    """特征筛选：按保留列取子集。"""
    idx = [input_idx[c] for c in output_columns]
    return np.asarray(X, dtype=np.float64)[:, idx]


# ======================================================================
# DeriveOp 内核（对照 _ops.py:362-395）
# ======================================================================
# 转译源码 → 可调用函数缓存（与 risk_ml/online_deploy/_ops.py 的 _DERIVE_FNS
# 对称，改动必须同步，由 test_scorer_parity 锁死两侧一致）。
_DERIVE_FNS = {}


def _get_derive_fn(source):
    """exec 转译源码并缓存（命名空间只给 np，白名单表达式安全）。"""
    fn = _DERIVE_FNS.get(source)
    if fn is None:
        ns = {}
        exec(source, {"np": np}, ns)  # 定义 _fd(X)
        fn = ns["_fd"]
        _DERIVE_FNS[source] = fn
    return fn


def kernel_derive(X, input_idx, output_columns, expressions):
    """特征衍生：表达式求值，返回原列 + 衍生列。

    expressions: [(target, source, variables)]
    - variables: 表达式变量，按 input_idx 取变量子数组（局部索引，与转译一致）
    - source: driver 侧 transpile 转译的 numpy 源码，定义 _fd(X_sub) -> (n,)
    """
    X = np.asarray(X, dtype=np.float64)
    derived = []
    for _target, source, variables in expressions:
        sub = X[:, [input_idx[v] for v in variables]]
        derived.append(_get_derive_fn(source)(sub))
    return np.column_stack([X] + derived)
