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
