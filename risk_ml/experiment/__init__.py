"""
实验模块 — 多配置风控建模实验对比

提供组合算子，在多个标签列、时间窗口、样本权重下自动跑 Optuna 调参，
汇总评估指标并选出最优实验。

公共 API:
- TimeWindow: 时间窗口配置
- ExperimentConfig: 单次实验配置
- ExperimentResult: 单次实验结果
- ExperimentRunner: 实验组合器（主类）
- make_experiment_grid: 便捷函数，生成笛卡尔积配置
- BaseMetric: 评估指标基类
- AUCMetric / KSMetric / LiftMetric: 内置指标
- DEFAULT_METRICS: 默认指标列表
"""

from .experiment_config import TimeWindow, ExperimentConfig, ExperimentResult
from .experiment_runner import ExperimentRunner
from .experiment_grid import make_experiment_grid
from .metrics import BaseMetric, AUCMetric, KSMetric, LiftMetric, DEFAULT_METRICS

__all__ = [
    "TimeWindow",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentRunner",
    "make_experiment_grid",
    "BaseMetric",
    "AUCMetric",
    "KSMetric",
    "LiftMetric",
    "DEFAULT_METRICS",
]
