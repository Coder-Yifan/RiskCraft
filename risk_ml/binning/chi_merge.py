"""
卡方分箱算子 — ChiMergeBinner

使用卡方检验（Chi-Square Test）进行自底向上的合并分箱，
是风控领域最常用的有监督分箱方法。

算法流程：
1. 将每个唯一值初始化为独立箱
2. 计算相邻箱的卡方统计量
3. 合并卡方值最小（最不显著）的相邻箱
4. 重复直到箱数 <= max_bins 且所有相邻箱的卡方值 >= 临界值
5. 检查最小箱占比约束
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .base_binner import BaseBinner


class ChiMergeBinner(BaseBinner):
    """
    卡方分箱算子：基于卡方检验的自底向上合并分箱。

    Parameters
    ----------
    max_bins : int, default=10
        最大分箱数。
    min_bins : int, default=2
        最小分箱数。
    bin_pct_threshold : float, default=0.05
        最小箱占比，低于此值的箱将被合并到相邻箱。
    confidence_level : float, default=0.9
        卡方检验的置信度，用于计算合并临界值。
    special_values : dict or None, default=None
        {col: [special_values]} 指定特殊值（如 -999 代表缺失），
        特殊值将被强制放入独立箱。
    categorical_features : list[str] or None, default=None
        分类特征列名列表，每列使用分类卡方分箱。

    Attributes
    ----------
    bin_edges_ : dict
        {col: np.ndarray} 各列的分箱边界点。
    bin_labels_ : dict
        {col: list[str]} 各列的箱标签。
    """

    def __init__(
        self,
        max_bins=10,
        min_bins=2,
        bin_pct_threshold=0.05,
        confidence_level=0.9,
        special_values=None,
        categorical_features=None,
    ):
        self.max_bins = max_bins
        self.min_bins = min_bins
        self.bin_pct_threshold = bin_pct_threshold
        self.confidence_level = confidence_level
        self.special_values = special_values
        self.categorical_features = categorical_features

    def _bin_column(self, x, y):
        """
        对单列执行卡方分箱。

        Args:
            x: np.ndarray，特征值
            y: np.ndarray，目标值（二分类 0/1）

        Returns:
            edges: np.ndarray，分箱边界
            labels: list[str]，箱标签
        """
        is_categorical = False  # 由 _bin_column 外层判断
        # 连续特征卡方分箱
        return self._chi_merge_continuous(x, y)

    def fit(self, X, y=None):
        """
        拟合卡方分箱。

        Args:
            X: pandas DataFrame
            y: 目标变量

        Returns:
            self
        """
        X = self._validate_data(X)
        if y is None:
            raise ValueError("卡方分箱需要目标变量 y")
        y = pd.Series(y, index=X.index)

        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]

        self.bin_edges_ = {}
        self.bin_labels_ = {}
        # 分类列映射信息（供 transform 使用）
        self._categorical_cols_ = set(self.categorical_features) if self.categorical_features else set()
        self._cat_maps_ = {}
        cat_features = self.categorical_features or []

        for col in X.columns:
            x_col = X[col].values
            if col in cat_features:
                edges, labels, cat_map = self._chi_merge_categorical(x_col, y.values)
                self._cat_maps_[col] = cat_map
            else:
                # 强制转为 float，避免 object 类型的 isnan 报错
                x_col = pd.to_numeric(x_col, errors="coerce").astype(float)
                edges, labels = self._chi_merge_continuous(x_col, y.values)
                cat_map = {}
            self.bin_edges_[col] = edges
            self.bin_labels_[col] = labels

        return self

    def _validate_data(self, X):
        """校验输入并返回 DataFrame"""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("输入必须为 pandas DataFrame")
        if X.shape[1] == 0:
            raise ValueError("DataFrame 没有列")
        return X

    # ------------------------------------------------------------------
    # 连续特征卡方分箱
    # ------------------------------------------------------------------

    def _chi_merge_continuous(self, x, y):
        """
        连续特征卡方分箱。

        算法：
        1. 排序去重，每个唯一值为一个初始箱
        2. 计算相邻箱的卡方值
        3. 合并卡方值最小的相邻箱
        4. 重复直到满足停止条件
        5. 应用最小箱占比约束
        """
        # 去除 NaN
        mask = ~np.isnan(x)
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) == 0:
            return np.array([-np.inf, np.inf]), ["(-inf, inf)"]

        # 初始化：每个唯一值一个箱
        sorted_vals = np.sort(np.unique(x_clean))
        if len(sorted_vals) <= 1:
            edges = np.array([-np.inf, np.inf])
            return edges, ["(-inf, inf)"]

        # 构建初始箱的边界和计数
        # 使用分位数作为初始切分点（避免唯一值过多）
        n_initial = min(len(sorted_vals), self.max_bins * 3)
        percentiles = np.linspace(0, 100, n_initial + 1)
        cut_points = np.percentile(sorted_vals, percentiles)
        cut_points = np.unique(cut_points)

        # 构建箱：统计每个箱的正负样本数
        bins = self._build_bins_from_cutpoints(x_clean, y_clean, cut_points)

        # 卡方合并
        chi2_threshold = chi2.ppf(self.confidence_level, df=1)

        while len(bins) > self.max_bins or (
            len(bins) > self.min_bins
            and self._min_chi2(bins) < chi2_threshold
        ):
            if len(bins) <= self.min_bins:
                break
            bins = self._merge_one(bins)

        # 最小箱占比约束
        bins = self._enforce_min_pct(bins, len(x_clean))

        # 生成边界和标签
        return self._bins_to_edges_labels(bins)

    def _build_bins_from_cutpoints(self, x, y, cut_points):
        """根据切分点构建箱的计数统计"""
        bins = []
        for i in range(len(cut_points) - 1):
            lower = cut_points[i]
            upper = cut_points[i + 1]
            if i == len(cut_points) - 2:
                mask = (x >= lower) & (x <= upper)
            else:
                mask = (x >= lower) & (x < upper)
            n_pos = int(y[mask].sum())
            n_neg = int(mask.sum() - n_pos)
            bins.append({
                "lower": lower,
                "upper": upper,
                "n_pos": n_pos,
                "n_neg": n_neg,
            })
        return bins

    def _compute_chi2(self, bin_a, bin_b):
        """计算两个相邻箱的卡方统计量"""
        total_a = bin_a["n_pos"] + bin_a["n_neg"]
        total_b = bin_b["n_pos"] + bin_b["n_neg"]
        total = total_a + total_b

        if total == 0:
            return 0.0

        # 期望频数
        pos_rate = (bin_a["n_pos"] + bin_b["n_pos"]) / total
        neg_rate = (bin_a["n_neg"] + bin_b["n_neg"]) / total

        e_a_pos = total_a * pos_rate
        e_a_neg = total_a * neg_rate
        e_b_pos = total_b * pos_rate
        e_b_neg = total_b * neg_rate

        chi2_val = 0.0
        for obs, exp in [
            (bin_a["n_pos"], e_a_pos),
            (bin_a["n_neg"], e_a_neg),
            (bin_b["n_pos"], e_b_pos),
            (bin_b["n_neg"], e_b_neg),
        ]:
            if exp > 0:
                chi2_val += (obs - exp) ** 2 / exp

        return chi2_val

    def _min_chi2(self, bins):
        """找到最小卡方值"""
        min_val = float("inf")
        for i in range(len(bins) - 1):
            chi2_val = self._compute_chi2(bins[i], bins[i + 1])
            min_val = min(min_val, chi2_val)
        return min_val

    def _merge_one(self, bins):
        """合并卡方值最小的一对相邻箱"""
        if len(bins) <= 1:
            return bins

        min_chi2 = float("inf")
        merge_idx = 0

        for i in range(len(bins) - 1):
            chi2_val = self._compute_chi2(bins[i], bins[i + 1])
            if chi2_val < min_chi2:
                min_chi2 = chi2_val
                merge_idx = i

        # 合并 merge_idx 和 merge_idx+1
        merged = {
            "lower": bins[merge_idx]["lower"],
            "upper": bins[merge_idx + 1]["upper"],
            "n_pos": bins[merge_idx]["n_pos"] + bins[merge_idx + 1]["n_pos"],
            "n_neg": bins[merge_idx]["n_neg"] + bins[merge_idx + 1]["n_neg"],
        }

        new_bins = bins[:merge_idx] + [merged] + bins[merge_idx + 2:]
        return new_bins

    def _enforce_min_pct(self, bins, total_count):
        """强制执行最小箱占比约束"""
        if total_count == 0:
            return bins

        changed = True
        while changed and len(bins) > self.min_bins:
            changed = False
            for i in range(len(bins)):
                bin_count = bins[i]["n_pos"] + bins[i]["n_neg"]
                if bin_count / total_count < self.bin_pct_threshold:
                    # 合并到相邻的较小卡方值箱
                    if i == 0:
                        merge_with = 0
                    elif i == len(bins) - 1:
                        merge_with = i - 1
                    else:
                        chi2_left = self._compute_chi2(bins[i - 1], bins[i])
                        chi2_right = self._compute_chi2(bins[i], bins[i + 1])
                        merge_with = i - 1 if chi2_left <= chi2_right else i

                    if merge_with == i:
                        merge_with = min(i, len(bins) - 2)

                    merged = {
                        "lower": min(bins[i]["lower"], bins[merge_with]["lower"]),
                        "upper": max(bins[i]["upper"], bins[merge_with]["upper"]),
                        "n_pos": bins[i]["n_pos"] + bins[merge_with]["n_pos"],
                        "n_neg": bins[i]["n_neg"] + bins[merge_with]["n_neg"],
                    }

                    idx_keep = min(i, merge_with)
                    bins = bins[:idx_keep] + [merged] + bins[idx_keep + 2:]
                    changed = True
                    break

        return bins

    def _bins_to_edges_labels(self, bins):
        """将箱结构转换为边界数组和标签列表"""
        edges = np.array([-np.inf] + [b["upper"] for b in bins[:-1]] + [np.inf])

        labels = []
        for i, b in enumerate(bins):
            if i == 0:
                labels.append(f"(-inf, {b['upper']:.4f}]")
            elif i == len(bins) - 1:
                labels.append(f"({b['lower']:.4f}, inf)")
            else:
                labels.append(f"({b['lower']:.4f}, {b['upper']:.4f}]")

        return edges, labels

    # ------------------------------------------------------------------
    # 分类特征卡方分箱
    # ------------------------------------------------------------------

    def _chi_merge_categorical(self, x, y):
        """
        分类特征卡方分箱。

        算法：
        1. 计算每个类别的正样本率
        2. 按正样本率排序
        3. 对排序后的类别执行卡方合并（同连续特征逻辑）

        Returns:
            edges, labels, cat_map
        """
        mask = ~pd.isna(x)
        x_clean = pd.Series(x[mask])
        y_clean = pd.Series(y[mask])

        if len(x_clean) == 0:
            cat_map = {}
            return np.array([-np.inf, 0, np.inf]), ["group_0", "group_1"], cat_map

        # 按正样本率排序
        categories = x_clean.unique()
        cat_rates = {}
        for cat in categories:
            cat_mask = x_clean == cat
            rate = y_clean[cat_mask].mean() if cat_mask.sum() > 0 else 0
            cat_rates[cat] = rate

        sorted_cats = sorted(cat_rates.keys(), key=lambda c: cat_rates[c])

        # 将排序后的类别映射为整数，然后使用连续分箱逻辑
        cat_map = {cat: i for i, cat in enumerate(sorted_cats)}
        x_mapped = x_clean.map(cat_map).values

        # 对映射后的整数执行连续分箱
        edges, labels = self._chi_merge_continuous(x_mapped, y_clean.values)

        return edges, labels, cat_map
