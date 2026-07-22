"""
实验配置模块 — TimeWindow / ExperimentConfig / ExperimentResult

- TimeWindow: 时间窗口值对象，按日期列筛选训练样本
- ExperimentConfig: 单次实验配置（标签列、时间窗口、权重列、流水线）
- ExperimentResult: 单次实验运行结果（指标、元数据、拟合估计器）
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class TimeWindow:
    """
    时间窗口配置，用于按日期列筛选训练样本。

    Parameters
    ----------
    date_column : str
        日期列名（如 "issue_d"），该列需可被 pd.to_datetime 解析。
    start_date : str
        起始日期（含），ISO 格式如 "2018-01-01"。
    end_date : str
        截止日期（含），ISO 格式如 "2018-03-31"。

    Example
    -------
    >>> tw = TimeWindow("issue_d", "2018-01-01", "2018-03-31")
    >>> mask = tw.filter(df)
    >>> df_sub = df[mask]
    """

    date_column: str
    start_date: str
    end_date: str

    def filter(self, X: pd.DataFrame) -> pd.Series:
        """
        返回布尔掩码，标识满足时间窗口的行。

        Args:
            X: 包含 date_column 的数据集

        Returns:
            布尔索引，True 表示该行在时间窗口内
        """
        dates = pd.to_datetime(X[self.date_column])
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        return (dates >= start) & (dates <= end)

    def __str__(self) -> str:
        return f"{self.start_date}~{self.end_date}"


@dataclass
class ExperimentConfig:
    """
    单次实验配置，定义一组建模参数变体。

    Parameters
    ----------
    name : str
        实验标识名（如 "baseline_30d"、"weighted_90d"）。
        在比较报告中作为行标识。
    label_col : str
        标签列名（如 "is_default_30d"、"is_default_90d"）。
        fit 时从输入 DataFrame 中提取此列作为 y。
    time_window : TimeWindow | None, default=None
        时间窗口配置。为 None 时使用全量数据。
    weight_col : str | None, default=None
        样本权重列名。为 None 时等权训练。
        权重值将从输入 DataFrame 中提取并传递给估计器的 fit()。
    pipeline : Any | None, default=None
        自定义流水线（sklearn Pipeline 或估计器实例）。
        为 None 时使用 ExperimentRunner 的默认流水线。
    fit_kwargs : dict | None, default=None
        传递给流水线 fit() 的额外参数。
        如 pipeline 是 sklearn Pipeline，sample_weight 路由
        为 ``{last_step_name}__sample_weight``。
    """

    name: str
    label_col: str
    time_window: TimeWindow | None = None
    weight_col: str | None = None
    pipeline: Any | None = None
    fit_kwargs: dict | None = None


@dataclass
class ExperimentResult:
    """
    单次实验运行结果。

    Attributes
    ----------
    config : ExperimentConfig
        原始实验配置。
    status : str
        运行状态："success" 或 "failed"。
    error : str | None
        失败时的错误信息。
    estimator : Any | None
        拟合后的最优流水线/估计器（OptunaTuner.best_estimator_）。
    best_params : dict | None
        Optuna 搜索的最优超参数。
    best_trial_score : float
        Optuna 最优 trial 的交叉验证评分。
    n_samples : int
        训练样本数（时间窗口过滤后）。
    n_features : int
        最终入模特征数。
    default_rate : float
        正样本比例。
    mean_iv : float
        入模特征的平均 IV 值。
    metric_values : dict
        {metric_name: value} 各评估指标的值，由 ExperimentRunner.metrics 动态决定。
    oot_metric_values : dict
        {metric_name: value} OOT 数据集上的评估指标值。
        仅当 ExperimentRunner 传入 oot 参数时填充。
    oot_n_samples : int
        OOT 数据集样本数。
    oot_default_rate : float
        OOT 数据集正样本比例。
    extra_label_metrics : dict
        {label_col: {metric_name: value}} 额外标签列的评估指标。
        仅当 ExperimentRunner 传入 eval_label_cols 参数时填充。
    oot_extra_label_metrics : dict
        {label_col: {metric_name: value}} OOT 数据集上额外标签列的评估指标。
        仅当同时传入 oot 和 eval_label_cols 时填充。
    training_time : float
        训练耗时（秒）。
    """

    config: ExperimentConfig
    status: str = "pending"
    error: str | None = None
    estimator: Any | None = None
    best_params: dict | None = None
    best_trial_score: float = 0.0
    n_samples: int = 0
    n_features: int = 0
    default_rate: float = 0.0
    mean_iv: float = 0.0
    metric_values: dict = field(default_factory=dict)
    oot_metric_values: dict = field(default_factory=dict)
    oot_n_samples: int = 0
    oot_default_rate: float = 0.0
    extra_label_metrics: dict = field(default_factory=dict)
    oot_extra_label_metrics: dict = field(default_factory=dict)
    training_time: float = 0.0
