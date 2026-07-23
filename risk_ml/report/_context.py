"""建模上下文与流水线属性提取。"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..experiment.metrics import BaseMetric, DEFAULT_METRICS


@dataclass
class PipelineAttributes:
    """从已拟合流水线中提取的属性汇总。

    通过 extract_pipeline_attributes() 自动填充，
    各字段可能为 None（流水线缺少对应步骤时）。
    """

    # --- FeatureCleaner ---
    drop_columns_: list[str] | None = None
    impute_values_: dict | None = None
    missing_threshold_: float | None = None
    variance_threshold_: float | None = None

    # --- BinnerWoeEncoder ---
    woe_map_: dict | None = None
    iv_values_bwe_: dict | None = None
    bin_edges_: dict | None = None
    bin_labels_: dict | None = None

    # --- IVSelector ---
    iv_values_: pd.Series | None = None
    iv_threshold_: float | None = None
    iv_max_iv_: float | None = None

    # --- CorrelationSelector ---
    correlation_matrix_: pd.DataFrame | None = None
    drop_features_: list[str] | None = None
    corr_threshold_: float | None = None

    # --- PSISelector ---
    psi_values_: pd.Series | None = None
    psi_threshold_: float | None = None

    # --- RiskXGBClassifier / 最终估计器 ---
    feature_importance_gain_: dict | None = None
    feature_importance_weight_: dict | None = None
    gain_total_: float | None = None
    weight_total_: float | None = None
    feature_names_in_: list[str] | None = None
    model_params_: dict | None = None
    n_features_in_: int | None = None


def extract_pipeline_attributes(pipeline: Any) -> PipelineAttributes:
    """从已拟合流水线提取属性。

    泛化 ExperimentRunner._extract_iv_values() 的 hasattr 遍历模式，
    遍历 Pipeline 各步骤或单个估计器，收集所有 post-fit 属性。

    Parameters
    ----------
    pipeline : Any
        sklearn Pipeline 或单估计器（已拟合）

    Returns
    -------
    PipelineAttributes
        提取的属性汇总
    """
    attrs = PipelineAttributes()

    # 获取步骤列表
    steps = []
    if hasattr(pipeline, "steps"):
        for _name, step in pipeline.steps:
            steps.append(step)
    else:
        steps.append(pipeline)

    for step in steps:
        # FeatureCleaner 属性
        if hasattr(step, "drop_columns_"):
            attrs.drop_columns_ = list(step.drop_columns_)
        if hasattr(step, "impute_values_"):
            attrs.impute_values_ = dict(step.impute_values_)
        if hasattr(step, "missing_threshold"):
            attrs.missing_threshold_ = step.missing_threshold
        if hasattr(step, "variance_threshold"):
            attrs.variance_threshold_ = step.variance_threshold

        # BinnerWoeEncoder 属性
        if hasattr(step, "woe_map_"):
            attrs.woe_map_ = step.woe_map_
        if hasattr(step, "iv_values_"):
            # 区分: BinnerWoeEncoder 的 iv_values_ 是 dict，
            # IVSelector 的 iv_values_ 是 pd.Series
            if isinstance(step.iv_values_, dict):
                attrs.iv_values_bwe_ = step.iv_values_
            elif isinstance(step.iv_values_, pd.Series):
                attrs.iv_values_ = step.iv_values_
        if hasattr(step, "bin_edges_"):
            attrs.bin_edges_ = step.bin_edges_
        if hasattr(step, "bin_labels_"):
            attrs.bin_labels_ = step.bin_labels_

        # IVSelector 属性
        if hasattr(step, "iv_threshold"):
            attrs.iv_threshold_ = step.iv_threshold
        if hasattr(step, "max_iv"):
            attrs.iv_max_iv_ = step.max_iv

        # CorrelationSelector 属性
        if hasattr(step, "correlation_matrix_"):
            attrs.correlation_matrix_ = step.correlation_matrix_
        if hasattr(step, "drop_features_"):
            attrs.drop_features_ = list(step.drop_features_)
        if hasattr(step, "corr_threshold"):
            attrs.corr_threshold_ = step.corr_threshold

        # PSISelector 属性
        if hasattr(step, "psi_values_") and isinstance(step.psi_values_, pd.Series):
            attrs.psi_values_ = step.psi_values_
        if hasattr(step, "psi_threshold"):
            attrs.psi_threshold_ = step.psi_threshold

        # 估计器属性（XGBClassifier 等，或 OptunaTuner 包装的分类器）
        # OptunaTuner: 属性在 best_estimator_ 上，而非 tuner 本身
        inner_step = step
        if hasattr(step, "best_estimator_") and hasattr(step, "study_"):
            # OptunaTuner: 从 best_estimator_ 提取
            inner_step = step.best_estimator_

        if hasattr(inner_step, "feature_importances_"):
            importances = inner_step.feature_importances_
            feature_names = getattr(inner_step, "feature_names_in_", None)
            if feature_names is not None:
                attrs.feature_importance_gain_ = dict(zip(feature_names, importances))
                attrs.gain_total_ = float(importances.sum())
        if hasattr(inner_step, "booster_"):
            # XGBoost: gain/weight importance
            try:
                booster = inner_step.booster_
                importance_gain = booster.get_score(importance_type="gain")
                importance_weight = booster.get_score(importance_type="weight")
                if importance_gain:
                    attrs.feature_importance_gain_ = importance_gain
                    attrs.gain_total_ = sum(importance_gain.values())
                if importance_weight:
                    attrs.feature_importance_weight_ = importance_weight
                    attrs.weight_total_ = sum(importance_weight.values())
            except Exception:
                pass
        if hasattr(inner_step, "feature_names_in_"):
            attrs.feature_names_in_ = list(inner_step.feature_names_in_)
            attrs.n_features_in_ = inner_step.n_features_in_

        # 模型参数: 优先取 Optuna best_params_, 否则取 get_params()
        if hasattr(step, "best_params_") and hasattr(step, "study_"):
            # OptunaTuner: 记录最优参数
            attrs.model_params_ = step.best_params_
        elif hasattr(inner_step, "get_params"):
            attrs.model_params_ = inner_step.get_params()

    return attrs


@dataclass
class ReportContext:
    """建模上下文: 包含所有报告算子需要的数据和元信息。

    构造时自动完成流水线属性提取和预测分数计算。
    所有字段可选，缺失时算子跳过或标注"未提供"。
    """

    # --- 元信息 ---
    model_name: str = ""
    developer: str = ""
    validator: str = ""
    business_owner: str = ""
    background: str = ""
    application: str = ""

    # --- 已拟合流水线 ---
    pipeline: Any = None

    # --- 数据集 ---
    X_train: pd.DataFrame | None = None
    y_train: np.ndarray | pd.Series | None = None
    X_test: pd.DataFrame | None = None
    y_test: np.ndarray | pd.Series | None = None
    X_oot: pd.DataFrame | None = None
    y_oot: np.ndarray | pd.Series | None = None

    # --- 预测分数（自动计算，也可预传入）---
    y_score_train: np.ndarray | None = None
    y_score_test: np.ndarray | None = None
    y_score_oot: np.ndarray | None = None

    # --- 评估指标 ---
    metrics: list[BaseMetric] | None = None

    # --- 标签定义 ---
    label_definition: dict = field(default_factory=lambda: {0: "好", 1: "坏"})

    # --- 对标模型分数（用于对比）---
    baseline_scores: dict | None = None  # {"train": ndarray, "test": ndarray, "oot": ndarray}

    # --- 灰样本 ---
    X_gray: pd.DataFrame | None = None
    y_gray: np.ndarray | pd.Series | None = None
    baseline_score_gray: np.ndarray | None = None

    # --- 特征元信息（外部提供）---
    feature_meta: dict | None = None  # {col: {"含义": ..., "来源": ..., "类别": ...}}

    # --- 样本来源分布（外部提供，用于1.4节）---
    sample_origin_distribution: pd.DataFrame | None = None
    dev_sample_origin_distribution: pd.DataFrame | None = None

    # --- 不同表现期数据（用于附件1.2）---
    mob_data: dict | None = None  # {"mob1": (y_true, y_score), ...}

    # --- 画像数据（用于附件1.3）---
    portrait_data: pd.DataFrame | None = None

    # --- 自动提取的流水线属性 ---
    pipeline_attrs: PipelineAttributes | None = None

    def __post_init__(self):
        """初始化后自动提取流水线属性和计算预测分数。"""
        if self.pipeline is not None and self.pipeline_attrs is None:
            self.pipeline_attrs = extract_pipeline_attributes(self.pipeline)
        if self.metrics is None:
            self.metrics = list(DEFAULT_METRICS)
        self._compute_scores()

    def _compute_scores(self):
        """从流水线自动计算预测分数（如未预传入）。"""
        if self.pipeline is None:
            return

        # 获取预测函数
        predict_fn = None
        if hasattr(self.pipeline, "predict_score"):
            predict_fn = self.pipeline.predict_score
        elif hasattr(self.pipeline, "predict_proba"):
            predict_fn = lambda X: self.pipeline.predict_proba(X)[:, 1]

        if predict_fn is None:
            return

        datasets = {
            "train": (self.X_train, self.y_score_train),
            "test": (self.X_test, self.y_score_test),
            "oot": (self.X_oot, self.y_score_oot),
        }

        for name, (X, y_score) in datasets.items():
            if y_score is None and X is not None:
                try:
                    score = predict_fn(X)
                    if name == "train":
                        self.y_score_train = score
                    elif name == "test":
                        self.y_score_test = score
                    elif name == "oot":
                        self.y_score_oot = score
                except Exception:
                    pass  # 预测失败则留 None
