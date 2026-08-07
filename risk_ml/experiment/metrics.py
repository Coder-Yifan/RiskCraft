"""
评估指标模块 — BaseMetric 及内置指标

提供可扩展的指标体系：
- BaseMetric: 抽象基类，定义 name / compute 标准接口
- AUCMetric: AUC (Area Under ROC Curve)
- KSMetric:  KS (Kolmogorov-Smirnov) 统计量
- LiftMetric: Lift 提升度（支持自定义分位数）

扩展方式：
    class GiniMetric(BaseMetric):
        name = "gini"
        def compute(self, y_true, y_score):
            from sklearn.metrics import roc_auc_score
            return 2 * roc_auc_score(y_true, y_score) - 1

    runner = ExperimentRunner(
        configs=configs,
        metrics=[AUCMetric(), KSMetric(), GiniMetric()],
    )
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseMetric(ABC):
    """
    评估指标抽象基类。

    所有自定义指标都应继承此类，实现 name / compute 接口。
    子类只需实现 compute(y_true, y_score) 即可，
    ExperimentRunner 会自动收集并汇总到 results_ 中。

    扩展示例：
        class GiniMetric(BaseMetric):
            name = "gini"
            def compute(self, y_true, y_score):
                from sklearn.metrics import roc_auc_score
                return 2 * roc_auc_score(y_true, y_score) - 1
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """指标名，用于 results_ 列名和日志输出，如 'auc'、'ks'"""
        ...

    @abstractmethod
    def compute(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """
        计算指标值。

        Args:
            y_true: 真实标签 (0/1)，shape (n_samples,)
            y_score: 正例预测概率，shape (n_samples,)

        Returns:
            指标值（float）
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class AUCMetric(BaseMetric):
    """
    AUC (Area Under ROC Curve) 指标。

    衡量模型区分正负样本的整体能力，值域 [0, 1]。
    AUC = 0.5 表示随机猜测，AUC = 1.0 表示完美区分。
    """

    @property
    def name(self) -> str:
        return "auc"

    def compute(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """
        计算 AUC。

        Args:
            y_true: 真实标签 (0/1)
            y_score: 正例预测概率

        Returns:
            AUC 值
        """
        from sklearn.metrics import roc_auc_score

        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        # 只有一个类别时无法计算 AUC
        if len(np.unique(y_true)) < 2:
            return 0.0

        return float(roc_auc_score(y_true, y_score))


class KSMetric(BaseMetric):
    """
    KS (Kolmogorov-Smirnov) 统计量 — 风控核心指标。

    衡量正负样本累积分布函数的最大差异，值域 [0, 1]。
    KS 越大，模型区分正负样本的能力越强。
    风控场景参考：KS > 0.2 可用，> 0.3 较好，> 0.4 优秀。
    """

    @property
    def name(self) -> str:
        return "ks"

    def compute(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """
        计算 KS 统计量。

        使用直方图 CDF 方法，与 OptunaTuner._ks_scorer 算法一致。

        Args:
            y_true: 真实标签 (0/1)
            y_score: 正例预测概率

        Returns:
            KS 值
        """
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        pos_score = y_score[y_true == 1]
        neg_score = y_score[y_true == 0]

        if len(pos_score) == 0 or len(neg_score) == 0:
            return 0.0

        # 概率输入保持 [0,1] 分箱（字节不变、零回归）；拉伸评分（非概率，如 300-900）
        # 按数据范围分箱——KS 对单调变换近似不变，保证拉伸分下 KS 仍正确
        lo, hi = float(np.min(y_score)), float(np.max(y_score))
        if lo == hi:
            return 0.0
        bins = np.linspace(0, 1, 101) if 0.0 <= lo and hi <= 1.0 else np.linspace(lo, hi, 101)
        pos_hist, _ = np.histogram(pos_score, bins=bins, density=True)
        neg_hist, _ = np.histogram(neg_score, bins=bins, density=True)

        pos_cdf = np.cumsum(pos_hist) / pos_hist.sum()
        neg_cdf = np.cumsum(neg_hist) / neg_hist.sum()

        return float(np.max(np.abs(pos_cdf - neg_cdf)))


class LiftMetric(BaseMetric):
    """
    Lift 提升度 — 风控常用指标。

    衡量模型在最高风险群体中识别正例的能力相对于随机猜测的提升倍数。
    例如 lift_10 = 3.0 表示前 10% 高分样本的正例率是全局的 3 倍。

    Parameters
    ----------
    percentile : float, default=10
        取分位数（%），如 10 表示前 10% 高分样本。
    """

    def __init__(self, percentile: float = 10):
        self.percentile = percentile
        self._name = f"lift_{int(percentile)}"

    @property
    def name(self) -> str:
        return self._name

    def compute(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """
        计算 Lift 提升度。

        Args:
            y_true: 真实标签 (0/1)
            y_score: 正例预测概率

        Returns:
            Lift 值
        """
        y_true = np.asarray(y_true, dtype=float)
        y_score = np.asarray(y_score, dtype=float)

        if len(y_true) == 0:
            return 0.0

        overall_rate = float(y_true.mean())
        if overall_rate == 0:
            return 0.0

        # 取前 percentile% 高分样本
        cutoff = np.percentile(y_score, 100 - self.percentile)
        top_mask = y_score >= cutoff

        if top_mask.sum() == 0:
            return 0.0

        top_rate = float(y_true[top_mask].mean())
        return top_rate / overall_rate


# 默认指标列表
DEFAULT_METRICS: list[BaseMetric] = [AUCMetric(), KSMetric(), LiftMetric(percentile=10)]
