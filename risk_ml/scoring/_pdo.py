"""
PDO 评分卡校准的 logit 拉伸算子 — PdoScoreScaler

风控从业人员习惯用评分卡参数而非 offset/factor：
- base_score : 基准分（如 600）
- base_p     : 基准分对应的坏样本概率（如 1/51 ≈ 0.0196，即 odds_good=50:1）
- pdo        : odds 每翻一倍分数增加的分值（points to double the odds，如 50）

换算：
    factor = pdo / ln(2)
    offset = base_score - factor·ln(odds_good)，odds_good = (1-base_p)/base_p

保证：score(base_p) == base_score，且 odds 每翻倍（风险减半）分数 +pdo。
换算后委托 LogitScoreScaler；base_score/base_p/pdo 保留供解释与 get_params。
"""

import numpy as np

from ._logit import LogitScoreScaler


class PdoScoreScaler(LogitScoreScaler):
    """PDO 评分卡校准的 logit 拉伸算子（子类化 LogitScoreScaler）。

    Parameters
    ----------
    base_score : float, default=600.0
        基准分（base_p 处对应的分数）。
    base_p : float, default=1/51
        基准分对应的坏样本概率（∈(0,1)）。默认 1/51 ≈ 0.0196（odds_good=50:1）。
    pdo : float, default=50.0
        odds 每翻一倍分数的增量（必须 > 0）。
    higher_is_safer : bool, default=True
        分数方向约定，见 LogitScoreScaler。

    Example
    -------
    >>> from risk_ml.scoring import PdoScoreScaler
    >>> scaler = PdoScoreScaler(base_score=600.0, base_p=1/51, pdo=50.0)
    >>> scaler([1/51, 1/101])   # odds 翻倍（1/51→1/101）→ 分数 +50
    array([600.        , 650.        ])
    """

    _name = "pdo"

    def __init__(self, base_score=600.0, base_p=1 / 51, pdo=50.0,
                 higher_is_safer=True):
        if not 0.0 < base_p < 1.0:
            raise ValueError(f"base_p 必须在 (0,1) 内，收到 {base_p!r}")
        if pdo <= 0:
            raise ValueError(f"pdo 必须 > 0，收到 {pdo!r}")
        odds_good = (1.0 - base_p) / base_p          # 好样本 odds = (1-p)/p
        factor = pdo / np.log(2.0)
        offset = base_score - factor * np.log(odds_good)
        super().__init__(
            offset=float(offset), factor=float(factor),
            higher_is_safer=higher_is_safer,
        )
        self.base_score = float(base_score)
        self.base_p = float(base_p)
        self.pdo = float(pdo)
