"""建模上下文与流水线属性提取。

核心设计变更:
- 单 DataFrame + tag_col + label_col 替代原 X_train/y_train/X_test/y_test/X_oot/y_oot
- ReportContext 辅助方法: get_datasets(), get_baseline_datasets(), get_gray_datasets()
- pipeline 自动计算预测分数写入 data[score_col]
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from risk_ml.experiment.metrics import BaseMetric, DEFAULT_METRICS


# ============================================================
# PipelineAttributes — 从已拟合流水线提取的属性汇总
# ============================================================

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

    遍历 Pipeline 各步骤或单个估计器，收集所有 post-fit 属性。
    支持 OptunaTuner（属性在 best_estimator_ 上）。

    Parameters
    ----------
    pipeline : Any
        sklearn Pipeline 或单估计器（已拟合）

    Returns
    -------
    PipelineAttributes
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

        # 估计器属性
        inner_step = step
        if hasattr(step, "best_estimator_") and hasattr(step, "study_"):
            inner_step = step.best_estimator_

        if hasattr(inner_step, "feature_importances_"):
            importances = inner_step.feature_importances_
            feature_names = getattr(inner_step, "feature_names_in_", None)
            if feature_names is not None:
                attrs.feature_importance_gain_ = dict(zip(feature_names, importances))
                attrs.gain_total_ = float(importances.sum())
        if hasattr(inner_step, "booster_"):
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

        # 模型参数
        if hasattr(step, "best_params_") and hasattr(step, "study_"):
            attrs.model_params_ = step.best_params_
        elif hasattr(inner_step, "get_params"):
            attrs.model_params_ = inner_step.get_params()

    # 兜底: 当 iv_values_ 为 None 但 iv_values_bwe_ 有值时，
    # 将 BinnerWoeEncoder 的 dict 转为 Series 填充到 iv_values_。
    if attrs.iv_values_ is None and attrs.iv_values_bwe_ is not None:
        attrs.iv_values_ = pd.Series(attrs.iv_values_bwe_)

    return attrs


# ============================================================
# tag → 中文名映射（get_datasets 系列方法使用）
# ============================================================

TAG_CN_MAP = {
    "train": "训练集",
    "test": "测试集",
    "oot": "跨时间验证集",
}


# ============================================================
# ReportContext — 建模上下文（单 DataFrame + tag 列）
# ============================================================

@dataclass
class ReportContext:
    """建模上下文: 包含所有报告算子需要的数据和元信息。

    核心设计: 传入一个完整的 DataFrame，通过 tag_col 区分 train/test/oot，
    通过 label_col 指定标签列名，通过 extra_labels 指定多标签列名。

    构造时自动完成流水线属性提取和预测分数计算。
    所有字段可选，缺失时算子产出占位表 + 提示文字。
    """

    # --- 核心数据（替代原 X_train/y_train/X_test/...） ---
    data: pd.DataFrame | None = None
    tag_col: str = "tag"
    label_col: str | None = None
    extra_labels: list[str] | None = None

    # --- 已拟合流水线 ---
    pipeline: Any = None

    # --- 预测分数 ---
    score_col: str | None = None

    # --- 对标模型 ---
    baseline_score_col: str | None = None

    # --- 灰样本 ---
    gray_tag: str | None = None  # 灰样本在 tag_col 中的值，如 "gray"

    # --- 时间列（用于月度拆分分析） ---
    time_col: str | None = None  # 时间列名，如 "transaction_time"，用于拆月分析

    # --- 元信息 ---
    model_name: str = ""
    developer: str = ""
    validator: str = ""
    business_owner: str = ""
    background: str = ""
    application: str = ""

    # --- 标签定义 ---
    label_definition: dict = field(default_factory=lambda: {0: "好", -1: "灰", 1: "坏"})
    observation_period: str = ""

    # --- 评估指标 ---
    metrics: list[BaseMetric] | None = None

    # --- 外部数据（无法放入 data 的特殊数据） ---
    sample_origin_distribution: pd.DataFrame | None = None
    sub_models: dict | None = None  # {"征信子": {"score_col": ..., "label_col": ...}}
    portrait_data: pd.DataFrame | None = None
    feature_meta: dict | None = None  # {col: {"含义": ..., "来源": ..., "类别": ...}}

    # --- 自动提取 ---
    pipeline_attrs: PipelineAttributes | None = None

    def __post_init__(self):
        """初始化后自动提取流水线属性和计算预测分数。"""
        if self.pipeline is not None and self.pipeline_attrs is None:
            self.pipeline_attrs = extract_pipeline_attributes(self.pipeline)
        if self.metrics is None:
            self.metrics = list(DEFAULT_METRICS)
        self._compute_scores()
        self._compute_iv_if_missing()

    # ---- 辅助方法: 从 tag 列拆分数据集 ----

    def get_datasets(
        self,
        label_col: str | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """获取各数据集的 (y_true, y_score) 字典。

        Parameters
        ----------
        label_col : str | None
            标签列名，默认使用 self.label_col

        Returns
        -------
        dict
            {"训练集": (y_true, y_score), "测试集": ..., "跨时间验证集": ...}
        """
        label = label_col or self.label_col
        if self.data is None or label is None or self.score_col is None:
            return {}

        result = {}
        for tag_val, cn_name in TAG_CN_MAP.items():
            mask = self.data[self.tag_col] == tag_val
            subset = self.data[mask]
            if len(subset) > 0 and label in subset.columns and self.score_col in subset.columns:
                y_true = subset[label].values.astype(float)
                y_score = subset[self.score_col].values.astype(float)
                result[cn_name] = (y_true, y_score)
        return result

    def get_baseline_datasets(
        self,
        label_col: str | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """获取各数据集的 (y_true, y_score_baseline) 字典。

        Parameters
        ----------
        label_col : str | None
            标签列名，默认使用 self.label_col

        Returns
        -------
        dict
            {"训练集": (y_true, baseline_score), ...}
        """
        label = label_col or self.label_col
        if self.data is None or label is None or self.baseline_score_col is None:
            return {}

        result = {}
        for tag_val, cn_name in TAG_CN_MAP.items():
            mask = self.data[self.tag_col] == tag_val
            subset = self.data[mask]
            if len(subset) > 0 and label in subset.columns and self.baseline_score_col in subset.columns:
                y_true = subset[label].values.astype(float)
                y_score = subset[self.baseline_score_col].values.astype(float)
                result[cn_name] = (y_true, y_score)
        return result

    def get_gray_datasets(
        self,
        label_col: str | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """获取灰样本数据集的 (y_true, y_score) 字典。

        灰样本通过 gray_tag 指定（如 "gray"），y_true 包含 Y=-1（灰）。

        Parameters
        ----------
        label_col : str | None
            标签列名

        Returns
        -------
        dict
            {"灰样本": (y_true, y_score)} 或与 train/test/oot 合并的数据集
        """
        label = label_col or self.label_col
        if self.data is None or label is None or self.gray_tag is None or self.score_col is None:
            return {}

        mask = self.data[self.tag_col] == self.gray_tag
        subset = self.data[mask]
        if len(subset) == 0 or label not in subset.columns or self.score_col not in subset.columns:
            return {}

        y_true = subset[label].values.astype(float)
        y_score = subset[self.score_col].values.astype(float)
        return {"灰样本": (y_true, y_score)}

    def get_datasets_with_gray(
        self,
        label_col: str | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """获取含灰样本的数据集字典（用于含灰报告）。

        将灰样本合并到各数据集的 y_true 中（灰样本 Y=-1）。

        Returns
        -------
        dict
            {"训练集(含灰)": (y_true, y_score), ...}
            y_true 中包含 Y=0(好), Y=1(坏), Y=-1(灰)
        """
        # 先获取不含灰的数据集
        datasets = self.get_datasets(label_col)
        gray = self.get_gray_datasets(label_col)

        if not gray:
            return datasets

        result = {}
        gray_y, gray_score = gray["灰样本"]

        for name, (y_true, y_score) in datasets.items():
            # 合并灰样本到训练集
            combined_y = np.concatenate([y_true, gray_y])
            combined_score = np.concatenate([y_score, gray_score])
            result[f"{name}(含灰)"] = (combined_y, combined_score)

        # 独立灰样本数据集
        result["灰样本"] = gray["灰样本"]
        return result

    def get_sample_stats(self, label_col: str | None = None) -> dict[str, dict]:
        """获取各数据集的样本统计。

        Returns
        -------
        dict
            {"训练集": {"goods": N, "bads": N, "gray": N, "total": N, "bad_rate": float}, ...}
        """
        from ._scoring import compute_sample_stats

        label = label_col or self.label_col
        if self.data is None or label is None:
            return {}

        result = {}
        all_tags = list(TAG_CN_MAP.keys())
        if self.gray_tag:
            all_tags.append(self.gray_tag)

        for tag_val in all_tags:
            cn_name = TAG_CN_MAP.get(tag_val, tag_val)
            mask = self.data[self.tag_col] == tag_val
            subset = self.data[mask]
            if len(subset) > 0 and label in subset.columns:
                result[cn_name] = compute_sample_stats(subset[label].values, self.label_definition)

        return result

    def get_monthly_datasets(
        self,
        label_col: str | None = None,
        tag_val: str | None = "oot",
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """获取按月拆分的数据集字典。

        将指定 tag 的数据按 time_col 的年月拆分，每月一个条目。
        tag_val=None 时遍历所有 tag，结果键名格式为 "数据集名-年月"。

        Parameters
        ----------
        label_col : str | None
            标签列名，默认使用 self.label_col
        tag_val : str | None
            要拆月的 tag 值，默认 "oot"；None 时全量数据集拆月

        Returns
        -------
        dict
            {"2026-04": (y_true, y_score), ...} 或 {"训练集-2024-01": ..., ...}
        """
        label = label_col or self.label_col
        if self.data is None or label is None or self.score_col is None or self.time_col is None:
            return {}

        if self.time_col not in self.data.columns:
            return {}

        result = {}

        if tag_val is not None:
            # 单个 tag 拆月
            tags_to_process = {tag_val: TAG_CN_MAP.get(tag_val, tag_val)}
        else:
            # 全量数据集拆月
            tags_to_process = TAG_CN_MAP.copy()

        for tv, cn_name in tags_to_process.items():
            mask = self.data[self.tag_col] == tv
            subset = self.data[mask]
            if len(subset) == 0:
                continue

            time_series = pd.to_datetime(subset[self.time_col])
            months = time_series.dt.to_period("M")

            for period in sorted(months.unique()):
                month_mask = months == period
                month_subset = subset[month_mask.values]
                if len(month_subset) > 0 and label in month_subset.columns and self.score_col in month_subset.columns:
                    y_true = month_subset[label].values.astype(float)
                    y_score = month_subset[self.score_col].values.astype(float)
                    key = f"{cn_name}-{period}" if tag_val is None else str(period)
                    result[key] = (y_true, y_score)

        return result

    def get_monthly_baseline_datasets(
        self,
        label_col: str | None = None,
        tag_val: str | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """获取按月拆分的对标模型数据集字典。

        Parameters
        ----------
        label_col : str | None
            标签列名
        tag_val : str | None
            要拆月的 tag 值，默认 "oot"

        Returns
        -------
        dict
            {"2026-04": (y_true, baseline_score), ...}
        """
        label = label_col or self.label_col
        tv = tag_val or "oot"
        if self.data is None or label is None or self.baseline_score_col is None or self.time_col is None:
            return {}

        if self.time_col not in self.data.columns:
            return {}

        mask = self.data[self.tag_col] == tv
        subset = self.data[mask]
        if len(subset) == 0:
            return {}

        time_series = pd.to_datetime(subset[self.time_col])
        months = time_series.dt.to_period("M")

        result = {}
        for period in sorted(months.unique()):
            month_mask = months == period
            month_subset = subset[month_mask.values]
            if len(month_subset) > 0 and label in month_subset.columns and self.baseline_score_col in month_subset.columns:
                y_true = month_subset[label].values.astype(float)
                y_score = month_subset[self.baseline_score_col].values.astype(float)
                result[str(period)] = (y_true, y_score)
        return result

    def _compute_scores(self):
        """从流水线自动计算预测分数，写入 data[score_col]。"""
        if self.pipeline is None or self.data is None:
            return
        if self.score_col is not None and self.score_col in self.data.columns:
            return  # 已有分数列

        # 自动计算
        predict_fn = getattr(self.pipeline, "predict_score", None)
        if predict_fn is None and hasattr(self.pipeline, "predict_proba"):
            predict_fn = lambda X: self.pipeline.predict_proba(X)[:, 1]
        if predict_fn is None:
            return

        # 确定特征列
        feature_cols = self._get_feature_columns()
        if not feature_cols:
            return

        self.score_col = "__y_score__"
        self.data[self.score_col] = np.nan
        for tag_val in TAG_CN_MAP.keys():
            mask = self.data[self.tag_col] == tag_val
            subset = self.data[mask]
            if len(subset) > 0:
                try:
                    self.data.loc[mask, self.score_col] = predict_fn(subset[feature_cols])
                except Exception:
                    pass

    def _get_feature_columns(self) -> list[str]:
        """获取特征列名列表。"""
        if self.pipeline is not None and hasattr(self.pipeline, "feature_names_in_"):
            return [c for c in self.pipeline.feature_names_in_ if c in self.data.columns]
        # 兜底：排除 tag/label/score/baseline 列
        exclude = {self.tag_col}
        if self.label_col:
            exclude.add(self.label_col)
        if self.score_col:
            exclude.add(self.score_col)
        if self.baseline_score_col:
            exclude.add(self.baseline_score_col)
        if self.extra_labels:
            exclude.update(self.extra_labels)
        return [c for c in self.data.columns if c not in exclude]

    def _compute_iv_if_missing(self):
        """当 pipeline 未产出 IV 时，从 data 自动计算（使用统一算法）。

        保证无论 IV 来自 BinnerWoeEncoder / IVSelector / 自动计算，
        结果都使用同一套 compute_iv_from_data 算法，数值一致。
        """
        if self.pipeline_attrs is not None and self.pipeline_attrs.iv_values_ is not None:
            return  # pipeline 已产出 IV，无需重复计算

        if self.data is None or self.label_col is None:
            return
        if self.label_col not in self.data.columns:
            return

        try:
            from ._scoring import compute_iv_from_data
            feature_cols = self._get_feature_columns()
            if not feature_cols:
                return

            # 只对 train 集计算 IV
            train_mask = self.data[self.tag_col] == "train"
            train_data = self.data.loc[train_mask, feature_cols]
            train_y = self.data.loc[train_mask, self.label_col]
            if len(train_data) > 0:
                if self.pipeline_attrs is None:
                    self.pipeline_attrs = PipelineAttributes()
                self.pipeline_attrs.iv_values_ = compute_iv_from_data(train_data, train_y)
        except Exception:
            pass
