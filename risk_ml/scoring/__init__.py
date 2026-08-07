"""
评分拉伸算子 — proba → 风险分

评分拉伸把模型输出的正例概率映射为风控风险分（风险领域通常不直接用概率值）。
算子化设计：新增拉伸方法只需继承 ``ScoreScaler`` 实现 ``transform(p)``，
RiskPipeline 通过 ``score_scaler`` 参数持有，在线部署通过 margin 折叠复用（零成本）。

用法:
    from risk_ml.scoring import LogitScoreScaler, PdoScoreScaler
    from risk_ml import RiskPipeline

    # 显式 offset/factor
    pipe = RiskPipeline(steps, score_scaler=LogitScoreScaler(offset=882.0, factor=72.13))
    # PDO 评分卡校准（600 分对应 odds 50:1，odds 翻倍 +50）
    pipe = RiskPipeline(steps, score_scaler=PdoScoreScaler(base_score=600, base_p=1/51, pdo=50))
    scores = pipe.predict_score(X)   # 默认（无 scaler）等价 predict_proba[:,1]
"""

from ._base import ScoreScaler
from ._logit import LogitScoreScaler
from ._pdo import PdoScoreScaler

__all__ = ["ScoreScaler", "LogitScoreScaler", "PdoScoreScaler"]
