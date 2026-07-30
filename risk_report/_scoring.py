"""分箱计算核心工具 — lift_table / swap / per_feature_ks / sample_stats。"""

import numpy as np
import pandas as pd


def compute_lift_table(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    n_bins: int = 10,
    baseline_score: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """模型分分箱表现表（Score Lift Table）。

    按分数等频分箱，计算每箱的好/坏/总量/总量%/坏率/KS/Lift/Cum_Lift。
    如提供 baseline_score，增加对标列并列对比。

    Parameters
    ----------
    y_true : array-like
        真实标签（0/1 二分类）
    y_score : array-like
        模型预测正例概率
    n_bins : int
        分箱数量，默认10
    baseline_score : array-like | None
        对标模型分数（可选），需与 y_true 长度一致

    Returns
    -------
    pd.DataFrame
        列: min, max, goods, bads, total, total%, bad_rate, ks, lift, cum_lift
        如有 baseline_score，增加 baseline_ 前缀列
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)

    # 等频分箱边界
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(y_score, quantiles)
    edges = np.unique(edges)  # 去重，可能少于 n_bins
    # 确保 -inf 和 inf 作为首尾
    edges[0] = -np.inf
    edges[-1] = np.inf

    # 分箱
    bin_indices = np.searchsorted(edges[1:-1], y_score, side="right")
    # bin_indices 现在是 0..n_actual_bins-1

    # 计算每箱统计
    rows = []
    cum_goods = 0
    cum_bads = 0
    total_goods = (y_true == 0).sum()
    total_bads = (y_true == 1).sum()
    overall_rate = total_bads / n if n > 0 else 0

    ks_values = []

    for i in range(len(edges) - 1):
        mask = bin_indices == i
        goods = int((y_true[mask] == 0).sum())
        bads = int((y_true[mask] == 1).sum())
        total = goods + bads
        cum_goods += goods
        cum_bads += bads

        total_pct = total / n if n > 0 else 0
        bad_rate = bads / total if total > 0 else 0
        lift = bad_rate / overall_rate if overall_rate > 0 else 0
        cum_lift = cum_bads / (cum_goods + cum_bads) / overall_rate if overall_rate > 0 and (cum_goods + cum_bads) > 0 else 0

        # KS: 累计坏率 - 累计好率
        cum_bad_pct = cum_bads / total_bads if total_bads > 0 else 0
        cum_good_pct = cum_goods / total_goods if total_goods > 0 else 0
        ks = cum_bad_pct - cum_good_pct
        ks_values.append(ks)

        rows.append({
            "min": edges[i],
            "max": edges[i + 1],
            "goods": goods,
            "bads": bads,
            "total": total,
            "total%": total_pct,
            "bad_rate": bad_rate,
            "ks": ks,
            "lift": lift,
            "cum_lift": cum_lift,
        })

    df = pd.DataFrame(rows)

    # baseline 对标列
    if baseline_score is not None:
        baseline_score = np.asarray(baseline_score)
        bl_indices = np.searchsorted(edges[1:-1], baseline_score, side="right")
        bl_rows = []
        bl_cum_goods = 0
        bl_cum_bads = 0

        for i in range(len(edges) - 1):
            mask = bl_indices == i
            goods = int((y_true[mask] == 0).sum())
            bads = int((y_true[mask] == 1).sum())
            total = goods + bads
            bl_cum_goods += goods
            bl_cum_bads += bads

            total_pct = total / n if n > 0 else 0
            bad_rate = bads / total if total > 0 else 0
            lift = bad_rate / overall_rate if overall_rate > 0 else 0
            cum_lift = bl_cum_bads / (bl_cum_goods + bl_cum_bads) / overall_rate if overall_rate > 0 and (bl_cum_goods + bl_cum_bads) > 0 else 0

            cum_bad_pct = bl_cum_bads / total_bads if total_bads > 0 else 0
            cum_good_pct = bl_cum_goods / total_goods if total_goods > 0 else 0
            ks = cum_bad_pct - cum_good_pct

            bl_rows.append({
                "baseline_min": edges[i],
                "baseline_max": edges[i + 1],
                "baseline_goods": goods,
                "baseline_bads": bads,
                "baseline_total": total,
                "baseline_total%": total_pct,
                "baseline_bad_rate": bad_rate,
                "baseline_ks": ks,
                "baseline_lift": lift,
                "baseline_cum_lift": cum_lift,
            })

        bl_df = pd.DataFrame(bl_rows)
        df = pd.concat([df, bl_df], axis=1)

    return df


def compute_swap_analysis(
    y_true: np.ndarray | pd.Series,
    y_score_new: np.ndarray | pd.Series,
    y_score_old: np.ndarray | pd.Series | None = None,
    cutoff_percentiles: list[float] = [10, 20],
) -> pd.DataFrame:
    """Swap In/Out 分析表。

    在指定切分比例下，计算新模型 vs 对标模型的拒绝/通过/swap。

    Parameters
    ----------
    y_true : array-like
        真实标签
    y_score_new : array-like
        新模型分数
    y_score_old : array-like | None
        对标模型分数（可选，无则仅展示新模型切分）
    cutoff_percentiles : list[float]
        切分百分比列表，默认 [10, 20]

    Returns
    -------
    pd.DataFrame
        列: 切分比例, 新总拒绝, 新通过, 新拒绝坏人, 新通过好人, ...
    """
    y_true = np.asarray(y_true)
    y_score_new = np.asarray(y_score_new)
    n = len(y_true)

    rows = []
    for pct in cutoff_percentiles:
        # 新模型 cutoff
        cutoff_new = np.percentile(y_score_new, pct)
        new_reject_mask = y_score_new <= cutoff_new
        new_pass_mask = ~new_reject_mask

        new_reject_total = int(new_reject_mask.sum())
        new_pass_total = int(new_pass_mask.sum())
        new_reject_bad = int((y_true[new_reject_mask] == 1).sum())
        new_pass_good = int((y_true[new_pass_mask] == 0).sum())

        row = {
            "切分比例": f"{pct}%",
            "新总拒绝": new_reject_total,
            "新通过": new_pass_total,
            "新拒绝坏人": new_reject_bad,
            "新通过好人": new_pass_good,
        }

        if y_score_old is not None:
            y_score_old = np.asarray(y_score_old)
            cutoff_old = np.percentile(y_score_old, pct)
            old_reject_mask = y_score_old <= cutoff_old

            # swap in: 老模型拒绝但新模型通过
            swap_in = int((old_reject_mask & new_pass_mask).sum())
            # swap out: 老模型通过但新模型拒绝
            swap_out = int((~old_reject_mask & new_reject_mask).sum())

            row.update({
                "对标总拒绝": int(old_reject_mask.sum()),
                "对标通过": int((~old_reject_mask).sum()),
                "swap_in": swap_in,
                "swap_out": swap_out,
                "净改善": swap_in - swap_out,
            })

        rows.append(row)

    return pd.DataFrame(rows)


def compute_per_feature_ks(
    X: pd.DataFrame,
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    n_bins: int = 10,
) -> pd.Series:
    """计算每个特征的 KS 值（单特征区分度）。

    对每个特征，按该特征值分箱后计算正负样本分布的 KS。

    Parameters
    ----------
    X : pd.DataFrame
        特征数据
    y_true : array-like
        真实标签（0/1）
    y_score : array-like
        模型分数（用于判断正负样本分组）
    n_bins : int
        分箱数

    Returns
    -------
    pd.Series
        每个特征的 KS 值，索引为特征名
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos_mask = y_score >= np.median(y_score)  # 高分段为"正"
    neg_mask = ~pos_mask

    # 确保 X 和 mask 对齐: 重置 X 的 index 以匹配数组位置
    X_aligned = X.reset_index(drop=True)
    y_true_aligned = y_true
    pos_mask_aligned = pos_mask
    neg_mask_aligned = neg_mask

    ks_dict = {}
    for col in X.columns:
        x = X_aligned[col].dropna()
        # 获取非缺失行的位置索引（整数位置）
        valid_positions = x.index.values  # dropna 后的整数位置
        pos_idx = pos_mask_aligned[valid_positions]
        neg_idx = neg_mask_aligned[valid_positions]

        if len(x) == 0 or pos_idx.sum() == 0 or neg_idx.sum() == 0:
            ks_dict[col] = 0.0
            continue

        # 等频分箱
        try:
            edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
            edges = np.unique(edges)
            edges[0] = -np.inf
            edges[-1] = np.inf

            pos_bins = np.searchsorted(edges[1:-1], x.values[pos_idx.values], side="right")
            neg_bins = np.searchsorted(edges[1:-1], x.values[neg_idx.values], side="right")

            # 计算每箱比例
            n_bins_actual = len(edges) - 1
            pos_dist = np.zeros(n_bins_actual)
            neg_dist = np.zeros(n_bins_actual)
            for b in range(n_bins_actual):
                pos_dist[b] = (pos_bins == b).sum()
                neg_dist[b] = (neg_bins == b).sum()

            pos_dist = pos_dist / pos_dist.sum() if pos_dist.sum() > 0 else pos_dist
            neg_dist = neg_dist / neg_dist.sum() if neg_dist.sum() > 0 else neg_dist

            ks = np.max(np.abs(np.cumsum(pos_dist) - np.cumsum(neg_dist)))
            ks_dict[col] = ks
        except Exception:
            ks_dict[col] = 0.0

    return pd.Series(ks_dict, index=X.columns)


def compute_sample_stats(
    y: np.ndarray | pd.Series,
    label_definition: dict | None = None,
) -> dict:
    """计算样本统计: 好/坏/灰/总量/坏占比。

    Parameters
    ----------
    y : array-like
        标签数组
    label_definition : dict | None
        标签含义映射，如 {0: "好", -1: "灰", 1: "坏"}
        默认按 0=好 / 1=坏 二分类

    Returns
    -------
    dict
        包含 goods, bads, gray, total, bad_rate 等键
    """
    y = np.asarray(y)
    if label_definition is None:
        label_definition = {0: "好", 1: "坏"}

    has_gray = -1 in label_definition

    goods = int((y == 0).sum())
    bads = int((y == 1).sum())
    gray = int((y == -1).sum()) if has_gray else 0
    total = goods + bads
    total_with_gray = goods + bads + gray
    bad_rate = bads / total if total > 0 else 0

    return {
        "goods": goods,
        "bads": bads,
        "gray": gray,
        "total": total,
        "total_with_gray": total_with_gray,
        "bad_rate": bad_rate,
    }


def compute_iv_from_data(
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    eps: float = 0.001,
) -> pd.Series:
    """从原始数据计算各特征的 IV 值（统一算法，高性能）。

    算法与 WoeEncoder / IVSelector 一致：
    - 对每列按唯一值分组（连续变量需先分箱再调用）
    - 计算每组的 dist_pos / dist_neg / WOE / IV 贡献
    - 求和得到特征级 IV

    性能优化:
    - numba 可用时: JIT 编译 + 多列并行（prange），200列×10万行 < 0.5s
    - numba 不可用时: numpy bincount 向量化方案，200列×10万行 < 2s

    Parameters
    ----------
    X : pd.DataFrame
        特征数据（建议已分箱或离散化，连续变量直接计算会因唯一值过多导致 IV 偏高）
    y : array-like
        目标变量（二分类 0/1）
    eps : float
        平滑因子，防止零频数导致除零

    Returns
    -------
    pd.Series
        各特征的 IV 值，索引为特征名
    """
    y_arr = np.asarray(y, dtype=np.float64)
    X_arr = X.values.astype(np.float64)
    total_pos = float(y_arr.sum())
    total_neg = float(len(y_arr) - total_pos)

    # 优先使用 numba 加速
    if _NUMBA_AVAILABLE:
        iv_arr = _iv_all_cols(X_arr, y_arr, total_pos, total_neg, eps)
    else:
        iv_arr = _iv_bincount(X_arr, y_arr, total_pos, total_neg, eps)

    return pd.Series(iv_arr, index=X.columns)


# ============================================================
# IV 计算加速: numba JIT + 并行
# ============================================================

try:
    from numba import njit, prange

    @njit
    def _iv_single_col(col_arr: np.ndarray, y_arr: np.ndarray,
                       total_pos: float, total_neg: float, eps: float) -> float:
        """单列 IV 计算（numba JIT）：排序 + 单遍扫描。

        避免 pandas 逐值 mask 的 O(n_unique × n) 复杂度，
        排序后单遍扫描为 O(n log n + n)。
        """
        n = len(col_arr)
        # 过滤 NaN（NaN != NaN）
        valid_count = 0
        for i in range(n):
            if col_arr[i] == col_arr[i]:
                valid_count += 1

        col_clean = np.empty(valid_count, dtype=np.float64)
        y_clean = np.empty(valid_count, dtype=np.float64)
        idx = 0
        for i in range(n):
            if col_arr[i] == col_arr[i]:
                col_clean[idx] = col_arr[i]
                y_clean[idx] = y_arr[i]
                idx += 1

        if valid_count == 0:
            return 0.0

        # 排序
        sort_idx = np.argsort(col_clean)
        sorted_col = col_clean[sort_idx]
        sorted_y = y_clean[sort_idx]

        # 第一遍: 计算唯一值数量
        n_groups = 0
        i = 0
        while i < valid_count:
            n_groups += 1
            val = sorted_col[i]
            while i < valid_count and sorted_col[i] == val:
                i += 1

        # 第二遍: 计算 IV
        iv = 0.0
        i = 0
        while i < valid_count:
            val = sorted_col[i]
            n_pos = 0.0
            n_neg = 0.0
            while i < valid_count and sorted_col[i] == val:
                if sorted_y[i] == 1.0:
                    n_pos += 1.0
                else:
                    n_neg += 1.0
                i += 1
            dist_pos = (n_pos + eps) / (total_pos + eps * n_groups)
            dist_neg = (n_neg + eps) / (total_neg + eps * n_groups)
            woe = np.log(dist_pos / dist_neg)
            iv += (dist_pos - dist_neg) * woe

        return iv

    @njit(parallel=True)
    def _iv_all_cols(X_arr: np.ndarray, y_arr: np.ndarray,
                     total_pos: float, total_neg: float, eps: float) -> np.ndarray:
        """多列并行 IV 计算（numba JIT + prange）。"""
        n_cols = X_arr.shape[1]
        result = np.empty(n_cols, dtype=np.float64)
        for j in prange(n_cols):
            result[j] = _iv_single_col(X_arr[:, j], y_arr, total_pos, total_neg, eps)
        return result

    # 触发 JIT 编译（小数据）
    _warmup_X = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    _warmup_y = np.array([0.0, 1.0, 0.0])
    _iv_all_cols(_warmup_X, _warmup_y, 1.0, 2.0, 0.001)
    del _warmup_X, _warmup_y

    _NUMBA_AVAILABLE = True

except ImportError:
    _NUMBA_AVAILABLE = False


def _iv_bincount(X_arr: np.ndarray, y_arr: np.ndarray,
                 total_pos: float, total_neg: float, eps: float) -> np.ndarray:
    """IV 计算回退方案: numpy bincount 向量化。

    当 numba 不可用时使用，比原始 Python 逐值 mask 快 50-100 倍。
    """
    n_cols = X_arr.shape[1]
    result = np.empty(n_cols, dtype=np.float64)

    for j in range(n_cols):
        col_arr = X_arr[:, j]
        valid = ~np.isnan(col_arr)
        col_clean = col_arr[valid]
        y_clean = y_arr[valid]

        groups = np.unique(col_clean)
        n_groups = len(groups)

        if n_groups == 0:
            result[j] = 0.0
            continue

        # 编码为 0-based index，用 bincount 一次性统计
        group_indices = np.searchsorted(groups, col_clean)
        pos_per_group = np.bincount(group_indices, weights=y_clean, minlength=n_groups)
        count_per_group = np.bincount(group_indices, minlength=n_groups).astype(np.float64)
        neg_per_group = count_per_group - pos_per_group

        dist_pos = (pos_per_group + eps) / (total_pos + eps * n_groups)
        dist_neg = (neg_per_group + eps) / (total_neg + eps * n_groups)
        woe = np.log(dist_pos / dist_neg)
        result[j] = float(np.sum((dist_pos - dist_neg) * woe))

    return result
