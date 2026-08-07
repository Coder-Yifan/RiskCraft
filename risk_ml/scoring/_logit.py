"""
显式 offset/factor 的 logit 评分拉伸算子 — LogitScoreScaler

标准评分卡公式：
    score = offset + scale·logit(p)，logit(p)=ln(p/(1-p))
- higher_is_safer=True  → scale = -factor（分数越高风险越低，标准约定）
- higher_is_safer=False → scale = +factor（少数内部评分）

部署端通过 ``offset / factor / higher_is_safer`` 属性直接做 margin 折叠
（logit(sigmoid(m)) ≡ m，score = offset + scale·m），无需先算 sigmoid 再算 logit。
"""

import numpy as np

from ._base import ScoreScaler


class LogitScoreScaler(ScoreScaler):
    """显式 offset/factor 的 logit 拉伸算子。

    Parameters
    ----------
    offset : float, default=600.0
        截距项。p=0.5（logit=0）时得分 = offset。
    factor : float, default=50.0
        斜率项。odds 每变化 e 倍，得分变化 |factor| 分。
    higher_is_safer : bool, default=True
        True → score = offset - factor·logit(p)（分数越高风险越低，标准评分卡）；
        False → score = offset + factor·logit(p)。

    Example
    -------
    >>> from risk_ml.scoring import LogitScoreScaler
    >>> scaler = LogitScoreScaler(offset=600.0, factor=50.0)
    >>> scaler([0.5, 0.9])          # p 越大（风险越高）分越低
    array([600.        , 490.05832265])
    """

    _name = "logit"

    def __init__(self, offset=600.0, factor=50.0, higher_is_safer=True):
        self.offset = float(offset)
        self.factor = float(factor)
        self.higher_is_safer = bool(higher_is_safer)

    def transform(self, p):
        p = np.clip(self._validate_p(p), 1e-15, 1 - 1e-15)
        # logit(p) = ln(p) - ln(1-p)，np.log1p(-p) 保持接近 1 时的精度
        logit = np.log(p) - np.log1p(-p)
        scale = -self.factor if self.higher_is_safer else self.factor
        return self.offset + scale * logit
