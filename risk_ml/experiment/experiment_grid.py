"""
实验网格生成器 — make_experiment_grid

根据标签列、时间窗口、权重列的笛卡尔积自动生成 ExperimentConfig 列表。
"""

import itertools
from typing import List

from .experiment_config import ExperimentConfig, TimeWindow


def make_experiment_grid(
    label_cols: List[str],
    time_windows: List[TimeWindow] | None = None,
    weight_cols: List[str] | None = None,
    name_prefix: str = "exp",
) -> List[ExperimentConfig]:
    """
    根据配置维度生成实验配置的笛卡尔积。

    Parameters
    ----------
    label_cols : list[str]
        标签列名列表，如 ["is_default_30d", "is_default_90d"]。
    time_windows : list[TimeWindow] | None, default=None
        时间窗口列表。为 None 时不按时间筛选（等价于 [None]）。
    weight_cols : list[str] | None, default=None
        样本权重列名列表。为 None 时等权训练（等价于 [None]）。
    name_prefix : str, default="exp"
        实验名前缀，自动编号。

    Returns
    -------
    list[ExperimentConfig]
        所有维度笛卡尔积生成的实验配置列表。

    Example
    -------
    >>> configs = make_experiment_grid(
    ...     label_cols=["is_default_30d", "is_default_90d"],
    ...     time_windows=[
    ...         TimeWindow("issue_d", "2018-01-01", "2018-03-31"),
    ...         TimeWindow("issue_d", "2018-04-01", "2018-06-30"),
    ...     ],
    ... )
    >>> len(configs)  # 2 labels * 2 windows * 1 weight = 4
    4
    """
    # None 替换为 [None] 以简化笛卡尔积逻辑
    _time_windows = time_windows if time_windows is not None else [None]
    _weight_cols = weight_cols if weight_cols is not None else [None]

    configs = []
    for i, (label, tw, wc) in enumerate(
        itertools.product(label_cols, _time_windows, _weight_cols)
    ):
        # 自动生成实验名：包含关键维度信息
        parts = [name_prefix, f"{i:03d}"]
        parts.append(label)
        if tw is not None:
            parts.append(f"{tw.start_date}~{tw.end_date}")
        if wc is not None:
            parts.append(f"w={wc}")
        name = "_".join(parts)

        configs.append(
            ExperimentConfig(
                name=name,
                label_col=label,
                time_window=tw,
                weight_col=wc,
            )
        )

    return configs


def make_feature_grid(
    feature_groups: List[List[str]],
    label_col: str,
    time_window: TimeWindow | None = None,
    weight_col: str | None = None,
    name_prefix: str = "feat",
) -> List[ExperimentConfig]:
    """根据特征组合列表生成实验配置。

    每组特征列名对应一个实验，适用于比较不同特征组合的建模效果。

    Parameters
    ----------
    feature_groups : list[list[str]]
        特征组合列表，每个子列表是一组特征列名。
        如 [["x1","x2","x3"], ["x1","x2","x3","x4","x5"]]。
    label_col : str
        标签列名。
    time_window : TimeWindow | None, default=None
        时间窗口配置，所有实验共用。
    weight_col : str | None, default=None
        样本权重列名，所有实验共用。
    name_prefix : str, default="feat"
        实验名前缀，自动编号并附加特征数。

    Returns
    -------
    list[ExperimentConfig]
        每组特征对应一个实验配置。

    Example
    -------
    >>> configs = make_feature_grid(
    ...     feature_groups=[
    ...         ["x1", "x2", "x3"],
    ...         ["x1", "x2", "x3", "x4", "x5"],
    ...     ],
    ...     label_col="is_fraud",
    ... )
    >>> len(configs)
    2
    >>> configs[0].feature_columns
    ['x1', 'x2', 'x3']
    """
    configs = []
    for i, features in enumerate(feature_groups):
        name = f"{name_prefix}_{i:03d}_{len(features)}feats"
        configs.append(
            ExperimentConfig(
                name=name,
                label_col=label_col,
                time_window=time_window,
                weight_col=weight_col,
                feature_columns=features,
            )
        )
    return configs
