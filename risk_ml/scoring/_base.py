"""
评分拉伸算子基类 — ScoreScaler

把正例概率 p ∈ [0,1] 映射为风控风险分（如 300-900 分）。无状态后处理算子，
供 RiskPipeline 打分（predict_score）与在线部署（margin 折叠）两端复用同一算子。

为什么不是 Pipeline step：sklearn Pipeline 的 step 是特征变换、最后一步是估计器，
对**模型输出**的拉伸不能作为普通 step 放在估计器之后。因此算子是 pipeline 持有的对象。

扩展方式：新增拉伸方法 = 继承 ScoreScaler 实现 transform(p) 即可，
pipeline 与部署端零改动。
"""

from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator


class ScoreScaler(BaseEstimator, ABC):
    """评分拉伸算子基类（无状态）。

    Parameters
    ----------
    无（子类定义自身参数）。

    Attributes
    ----------
    offset : float
        截距项（标准评分卡约定，如 600）。
    factor : float
        斜率项（如 50/ln2 ≈ 72.13）。
    higher_is_safer : bool
        True 表示分数越高风险越低（标准约定），False 表示分数越高风险越高。

    Example
    -------
    >>> from risk_ml.scoring import LogitScoreScaler
    >>> scaler = LogitScoreScaler(offset=600.0, factor=50.0)
    >>> scaler.transform([0.5, 0.9])   # p=0.5 → 600；p=0.9（高风险）→ 更低分
    array([600.        , 490.05832265])
    """

    _name = "base"

    def fit(self, X=None, y=None):
        """无状态算子：直接返回 self。"""
        return self

    @abstractmethod
    def transform(self, p):
        """把形状 (n,) 的正例概率数组映射为风险分数组 (n,)。"""
        raise NotImplementedError("子类必须实现 transform(p)")

    def __call__(self, p):
        """运算符语法：scaler(p) 等价于 scaler.transform(p)。"""
        return self.transform(p)

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _validate_p(self, p):
        """校验并转换为 float64 一维数组。"""
        arr = np.asarray(p, dtype=float)
        if arr.ndim != 1:
            raise ValueError(
                f"输入必须是形状 (n,) 的一维概率数组，收到 {arr.shape}"
            )
        if np.any(arr < 0.0) or np.any(arr > 1.0):
            raise ValueError("概率值必须在 [0, 1] 内")
        return arr

    def _more_tags(self):
        """sklearn 标签：二分类后处理算子，无需 fit。"""
        return {"binary_only": True, "requires_fit": False}
