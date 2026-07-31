"""
实验组合器 — ExperimentRunner

在多个实验配置下自动运行风控建模流水线：
1. 按配置提取标签列、筛选时间窗口、提取样本权重
2. 用 OptunaTuner 对每个配置进行贝叶斯超参搜索
3. 计算用户指定的评估指标（AUC / KS / Lift 等）
4. 汇总为比较报告（results_），选出最优实验

设计参照 OptunaTuner 的组合估计器模式。
"""

import time
from typing import Any, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline

from .._base import validate_dataframe
from .._pipeline import RiskPipeline
from ..estimator.optuna_tuner import OptunaTuner
from .experiment_config import ExperimentConfig, ExperimentResult, TimeWindow
from .metrics import BaseMetric, DEFAULT_METRICS


class ExperimentRunner(BaseEstimator):
    """
    实验组合器：在多个配置下运行风控建模流水线并比较结果。

    接受一组 ExperimentConfig，对每个配置：
    1. 从输入 DataFrame 提取指定标签列作为 y
    2. 按时间窗口筛选样本（如配置）
    3. 提取样本权重（如配置）
    4. 用 OptunaTuner 进行贝叶斯超参搜索
    5. 计算评估指标（AUC / KS / Lift 等，用户可扩展）
    6. 汇总入模特征数、平均 IV 等信息

    运行完毕后，产生比较报告（results_）并暴露最优实验的估计器。

    Parameters
    ----------
    configs : list[ExperimentConfig]
        实验配置列表。
    pipeline : Any | None, default=None
        默认流水线（sklearn Pipeline 或估计器实例）。
        各 config 未指定 pipeline 时使用此流水线。
        为 None 时使用风控建模标准流水线。
    feature_columns : list[str] | None, default=None
        特征列名列表。为 None 时自动推断（排除所有 config 中的
        标签列、日期列、权重列）。
    metrics : list[BaseMetric] | None, default=None
        评估指标列表。为 None 时使用默认指标（AUC + KS + Lift@10%）。
        用户可传入自定义指标实例，结果报告中自动出现对应列。
    scoring : str, default="ks"
        Optuna 调参的优化目标，同时决定 best 选举的排序依据。
        支持 "ks" / "roc_auc" / "f1" 等。
    n_trials : int, default=30
        每个实验的 Optuna 搜索轮数。
    tuner_cv : int, default=3
        Optuna 内部 cross_val_score 折数（3 折更快）。
    oot : pd.DataFrame | None, default=None
        OOT（Out-Of-Time）跨时间验证数据集。
        为 None 时指标在训练集上计算；传入后指标在 OOT 数据上计算，
        results_ 中新增 oot_ 前缀列。模型仍在训练集上 fit。
    eval_label_cols : list[str] | None, default=None
        额外评估标签列名列表。这些标签不参与训练，仅用于评估。
        传入后 results_ 中新增 {label}_{metric} 列，
        若同时传入 oot，新增 oot_{label}_{metric} 列。
    n_jobs : int, default=1
        实验并行数（使用 joblib）。
    random_state : int | None, default=42
        随机种子。
    verbose : int, default=1
        日志级别。0=静默，1=进度，2=详细。

    Attributes
    ----------
    results_ : pd.DataFrame
        所有实验的比较报告，每行一个实验。
    best_config_ : ExperimentConfig
        最优实验配置（按 scoring 指标降序）。
    best_estimator_ : Any
        最优实验的已拟合流水线（已调参）。
    best_score_ : float
        最优实验的指定评分指标值。
    experiments_ : dict[str, ExperimentResult]
        {config.name: ExperimentResult} 实验结果字典。

    Example
    -------
    >>> from risk_ml.experiment import (
    ...     ExperimentRunner, ExperimentConfig, TimeWindow, make_experiment_grid
    ... )
    >>> configs = make_experiment_grid(
    ...     label_cols=["is_default_30d", "is_default_90d"],
    ...     time_windows=[
    ...         TimeWindow("issue_d", "2018-01-01", "2018-03-31"),
    ...         TimeWindow("issue_d", "2018-04-01", "2018-06-30"),
    ...     ],
    ... )
    >>> runner = ExperimentRunner(configs=configs, scoring="ks", n_trials=30)
    >>> runner.fit(df)
    >>> print(runner.results_)
    >>> y_score = runner.predict_score(X_test)
    """

    def __init__(
        self,
        configs: List[ExperimentConfig],
        pipeline: Any | None = None,
        feature_columns: List[str] | None = None,
        metrics: List[BaseMetric] | None = None,
        scoring: str = "ks",
        n_trials: int = 30,
        tuner_cv: int = 3,
        oot: pd.DataFrame | None = None,
        eval_label_cols: List[str] | None = None,
        n_jobs: int = 1,
        random_state: int | None = 42,
        verbose: int = 1,
    ):
        self.configs = configs
        self.pipeline = pipeline
        self.feature_columns = feature_columns
        self.metrics = metrics
        self.scoring = scoring
        self.n_trials = n_trials
        self.tuner_cv = tuner_cv
        self.oot = oot
        self.eval_label_cols = eval_label_cols
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        """
        运行所有实验配置。

        Parameters
        ----------
        X : pd.DataFrame
            输入数据，包含特征列、标签列、日期列（可选）、权重列（可选）。
        y : ignored
            标签从各 ExperimentConfig.label_col 提取，此参数保留 sklearn 兼容性。

        Returns
        -------
        self
        """
        X = validate_dataframe(X)

        # 解析指标列表
        metrics = self.metrics if self.metrics is not None else DEFAULT_METRICS

        # 解析特征列（runner 级默认）
        default_feature_columns = self._resolve_feature_columns(X)

        # 运行每个实验
        self.experiments_ = {}
        for i, config in enumerate(self.configs):
            if self.verbose >= 1:
                print(
                    f"[ExperimentRunner] 运行实验 {i + 1}/{len(self.configs)}: "
                    f"{config.name}"
                )

            # config 级 feature_columns 优先于 runner 级
            feature_columns = config.feature_columns or default_feature_columns
            result = self._run_single(config, X, feature_columns, metrics)
            self.experiments_[config.name] = result

        # 汇总结果
        self.results_ = self._build_results_dataframe(metrics)

        # 选举最优
        self._select_best(metrics)

        return self

    # ------------------------------------------------------------------
    # predict（委托最优估计器）
    # ------------------------------------------------------------------

    def predict(self, X):
        """使用最优估计器预测类别标签。"""
        self._check_is_fitted()
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        """使用最优估计器预测类别概率。"""
        self._check_is_fitted()
        return self.best_estimator_.predict_proba(X)

    def predict_score(self, X):
        """使用最优估计器预测正例概率（风控评分）。"""
        return self.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _resolve_feature_columns(self, X: pd.DataFrame) -> List[str]:
        """
        确定特征列名列表。

        优先使用 self.feature_columns，否则自动推断：
        1. 排除所有 config 中的 label_col、time_window.date_column、weight_col
        2. 排除 datetime 类型的列（不能作为数值特征）
        """
        if self.feature_columns is not None:
            return self.feature_columns

        # 收集需要排除的元数据列
        exclude_cols = set()
        for config in self.configs:
            exclude_cols.add(config.label_col)
            if config.time_window is not None:
                exclude_cols.add(config.time_window.date_column)
            if config.weight_col is not None:
                exclude_cols.add(config.weight_col)
        # 排除 eval_label_cols（仅评估用，不作为特征）
        if self.eval_label_cols is not None:
            exclude_cols.update(self.eval_label_cols)

        # 排除 datetime 类型列
        feature_cols = []
        for c in X.columns:
            if c in exclude_cols:
                continue
            if pd.api.types.is_datetime64_any_dtype(X[c]):
                continue
            feature_cols.append(c)

        return feature_cols

    def _resolve_pipeline(self, config: ExperimentConfig) -> Any:
        """
        获取实验使用的流水线。

        config.pipeline 优先，否则使用 self.pipeline，
        都为 None 则构建默认风控流水线。
        """
        if config.pipeline is not None:
            return config.pipeline
        if self.pipeline is not None:
            return self.pipeline
        return self._build_default_pipeline()

    def _build_pipeline_search_space(self, pipe: Any) -> dict | None:
        """
        为 Pipeline 生成兼容的 Optuna 搜索空间。

        OptunaTuner 默认搜索空间是单估计器参数，传入 Pipeline 时
        需要加 `step__` 前缀，否则 set_params 会报错。

        如果估计器不是 Pipeline，返回 None（使用 OptunaTuner 默认空间）。
        """
        if not isinstance(pipe, Pipeline):
            return None

        # 找到分类器步骤（最后一个步骤）
        last_step_name, last_step = pipe.steps[-1]

        # 从 OptunaTuner 复制默认搜索空间，加前缀
        from ..estimator.optuna_tuner import _DEFAULT_SEARCH_SPACE
        return {
            f"{last_step_name}__{param}": bounds
            for param, bounds in _DEFAULT_SEARCH_SPACE.items()
            if hasattr(last_step, param)  # 只包含该步骤实际支持的参数
        }

    def _build_default_pipeline(self) -> RiskPipeline:
        """
        构建风控建模标准流水线。

        FeatureCleaner → BinnerWoeEncoder → IVSelector →
        CorrelationSelector → RiskXGBClassifier
        """
        from ..preprocessing import FeatureCleaner
        from ..encoding import BinnerWoeEncoder
        from ..feature_selection import IVSelector, CorrelationSelector
        from ..estimator import RiskXGBClassifier

        return RiskPipeline([
            ("cleaner", FeatureCleaner()),
            ("binner_woe", BinnerWoeEncoder()),
            ("iv_selector", IVSelector()),
            ("corr_selector", CorrelationSelector()),
            ("classifier", RiskXGBClassifier()),
        ])

    def _prepare_fit_kwargs(
        self, config: ExperimentConfig, X_full: pd.DataFrame, sub_index: pd.Index
    ) -> dict:
        """
        根据实验配置准备 fit 参数（主要是 sample_weight 路由）。
        """
        fit_kwargs = dict(config.fit_kwargs) if config.fit_kwargs else {}

        if config.weight_col is not None and "sample_weight" not in str(fit_kwargs):
            weights = X_full.loc[sub_index, config.weight_col].values

            pipe = self._resolve_pipeline(config)
            if isinstance(pipe, Pipeline):
                # sklearn Pipeline: 路由到最后一步
                last_step_name = pipe.steps[-1][0]
                fit_kwargs[f"{last_step_name}__sample_weight"] = weights
            else:
                # 单估计器: 直接传递
                fit_kwargs["sample_weight"] = weights

        return fit_kwargs

    def _extract_iv_values(self, estimator: Any):
        """从已拟合的流水线中提取 IV 值"""
        if hasattr(estimator, "iv_values_"):
            return estimator.iv_values_

        # 遍历 Pipeline 各步骤
        if hasattr(estimator, "steps"):
            for _name, step in estimator.steps:
                if hasattr(step, "iv_values_"):
                    return step.iv_values_

        return None

    def _extract_n_features(self, estimator: Any) -> int:
        """从已拟合的流水线中提取最终入模特征数"""
        # 优先从分类器获取
        if hasattr(estimator, "n_features_in_"):
            return estimator.n_features_in_

        # Pipeline: 从最后一步获取
        if hasattr(estimator, "steps"):
            for _name, step in reversed(estimator.steps):
                if hasattr(step, "n_features_in_"):
                    return step.n_features_in_

        return 0

    def _run_single(
        self,
        config: ExperimentConfig,
        X_full: pd.DataFrame,
        feature_columns: List[str],
        metrics: List[BaseMetric],
    ) -> ExperimentResult:
        """运行单个实验配置，返回 ExperimentResult。"""
        result = ExperimentResult(config=config)
        start_time = time.time()

        try:
            # 1. 提取 y
            if config.label_col not in X_full.columns:
                raise ValueError(
                    f"标签列 '{config.label_col}' 不在输入数据中。"
                    f" 可用列: {list(X_full.columns)}"
                )
            y_cfg = X_full[config.label_col].values

            # 2. 时间窗口过滤
            if config.time_window is not None:
                mask = config.time_window.filter(X_full)
                X_features = X_full.loc[mask, feature_columns]
                y_sub = y_cfg[mask.values]
            else:
                X_features = X_full[feature_columns]
                y_sub = y_cfg

            # 3. 准备 fit_kwargs（含 sample_weight 路由）
            fit_kwargs = self._prepare_fit_kwargs(config, X_full, X_features.index)

            # 4. 获取流水线并 clone
            pipe = clone(self._resolve_pipeline(config))

            # 4.5 准备验证集数据（holdout 评估）
            X_val_features = None
            y_val_sub = None
            if self.oot is not None:
                X_val_features = self.oot[feature_columns]
                if config.label_col in self.oot.columns:
                    y_val_sub = self.oot[config.label_col].values

            # 5. 构造 OptunaTuner 并调参
            # Pipeline 需要 step__param 格式的搜索空间
            search_space = self._build_pipeline_search_space(pipe)

            tuner = OptunaTuner(
                estimator=pipe,
                n_trials=self.n_trials,
                search_space=search_space,
                scoring=self.scoring,
                cv=self.tuner_cv,
                n_jobs=1,  # 每个实验内部单线程，并行由外层控制
                random_state=self.random_state,
                verbose=max(0, self.verbose - 1),  # 内层降一级日志
            )

            # 有验证集时使用 holdout 评估，否则使用 CV
            if X_val_features is not None and y_val_sub is not None:
                tuner.fit(X_features, y_sub, X_val=X_val_features, y_val=y_val_sub, **fit_kwargs)
            else:
                tuner.fit(X_features, y_sub, **fit_kwargs)

            # 6. 获取最优估计器
            best_estimator = tuner.best_estimator_

            # 7. 计算训练集指标
            # 尝试 predict_score，否则用 predict_proba
            if hasattr(best_estimator, "predict_score"):
                y_score = best_estimator.predict_score(X_features)
            else:
                y_score = best_estimator.predict_proba(X_features)[:, 1]

            for metric in metrics:
                result.metric_values[metric.name] = metric.compute(y_sub, y_score)

            # 8. 额外标签列评估（训练集）
            if self.eval_label_cols is not None:
                for label in self.eval_label_cols:
                    if label not in X_full.columns:
                        continue
                    y_eval = X_full.loc[X_features.index, label].values
                    result.extra_label_metrics[label] = {
                        m.name: m.compute(y_eval, y_score) for m in metrics
                    }

            # 9. OOT 数据集评估
            if self.oot is not None:
                X_oot_features = self.oot[feature_columns]
                # OOT 标签
                if config.label_col in self.oot.columns:
                    y_oot = self.oot[config.label_col].values
                else:
                    y_oot = None

                # OOT 预测分数（只需算一次）
                if hasattr(best_estimator, "predict_score"):
                    y_score_oot = best_estimator.predict_score(X_oot_features)
                else:
                    y_score_oot = best_estimator.predict_proba(X_oot_features)[:, 1]

                # config.label_col 的 OOT 指标
                if y_oot is not None:
                    for metric in metrics:
                        result.oot_metric_values[metric.name] = metric.compute(
                            y_oot, y_score_oot
                        )
                    result.oot_n_samples = len(y_oot)
                    result.oot_default_rate = float(y_oot.mean())

                # OOT + eval_label_cols 组合
                if self.eval_label_cols is not None:
                    for label in self.eval_label_cols:
                        if label not in self.oot.columns:
                            continue
                        y_oot_label = self.oot[label].values
                        result.oot_extra_label_metrics[label] = {
                            m.name: m.compute(y_oot_label, y_score_oot) for m in metrics
                        }

            # 10. 记录 Optuna 搜索结果
            result.best_params = tuner.best_params_
            result.best_trial_score = tuner.best_score_
            result.estimator = best_estimator

            # 11. 提取元数据
            result.n_samples = len(y_sub)
            result.default_rate = float(y_sub.mean())
            result.n_features = self._extract_n_features(best_estimator)

            iv_values = self._extract_iv_values(best_estimator)
            if iv_values is not None:
                if hasattr(iv_values, "mean"):
                    result.mean_iv = float(iv_values.mean())
                elif isinstance(iv_values, dict):
                    result.mean_iv = float(np.mean(list(iv_values.values())))
                else:
                    result.mean_iv = float(np.mean(list(iv_values)))

            result.status = "success"

        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            if self.verbose >= 1:
                print(f"[ExperimentRunner] 实验失败: {config.name} — {e}")

        result.training_time = time.time() - start_time
        return result

    def _build_results_dataframe(self, metrics: List[BaseMetric]) -> pd.DataFrame:
        """将所有实验结果汇总为 DataFrame"""
        rows = []
        for result in self.experiments_.values():
            row = {
                "name": result.config.name,
                "label_col": result.config.label_col,
                "time_window": (
                    str(result.config.time_window)
                    if result.config.time_window
                    else "all"
                ),
                "weight_col": result.config.weight_col or "none",
                "status": result.status,
                "n_samples": result.n_samples,
                "default_rate": result.default_rate,
                "n_features": result.n_features,
                "mean_iv": result.mean_iv,
                "best_trial_score": result.best_trial_score,
                "training_time": result.training_time,
                "error": result.error or "",
            }
            # 训练集动态指标列
            for metric in metrics:
                row[metric.name] = result.metric_values.get(metric.name, np.nan)

            # OOT 指标列
            if self.oot is not None:
                row["oot_n_samples"] = result.oot_n_samples
                row["oot_default_rate"] = result.oot_default_rate
                for metric in metrics:
                    row[f"oot_{metric.name}"] = result.oot_metric_values.get(
                        metric.name, np.nan
                    )

            # 额外标签列（训练集）
            if self.eval_label_cols is not None:
                for label in self.eval_label_cols:
                    label_metrics = result.extra_label_metrics.get(label, {})
                    for metric in metrics:
                        row[f"{label}_{metric.name}"] = label_metrics.get(
                            metric.name, np.nan
                        )

            # OOT + 额外标签列
            if self.oot is not None and self.eval_label_cols is not None:
                for label in self.eval_label_cols:
                    oot_label_metrics = result.oot_extra_label_metrics.get(label, {})
                    for metric in metrics:
                        row[f"oot_{label}_{metric.name}"] = oot_label_metrics.get(
                            metric.name, np.nan
                        )

            rows.append(row)

        return pd.DataFrame(rows)

    def _select_best(self, metrics: List[BaseMetric]):
        """选举最优实验（优先从 OOT 指标选举）"""
        # 确定 scoring 对应的列名
        score_col = self._scoring_to_col(metrics)

        # 若有 OOT，优先从 OOT 指标选举
        if self.oot is not None:
            oot_col = f"oot_{score_col}"
            if oot_col in self.results_.columns:
                score_col = oot_col

        successful = self.results_[self.results_["status"] == "success"]
        if successful.empty:
            raise RuntimeError("所有实验均失败，无法选举最优")

        best_idx = successful[score_col].idxmax()
        best_name = self.results_.loc[best_idx, "name"]

        best_result = self.experiments_[best_name]
        self.best_config_ = best_result.config
        self.best_estimator_ = best_result.estimator
        self.best_score_ = best_result.metric_values.get(score_col, 0.0)

        if self.verbose >= 1:
            print(
                f"[ExperimentRunner] 最优实验: {best_name} "
                f"({score_col}={self.best_score_:.4f})"
            )

    def _scoring_to_col(self, metrics: List[BaseMetric]) -> str:
        """
        将 scoring 参数映射到 results_ 中的列名。

        例如 "ks" → "ks"，"roc_auc" → "auc"
        """
        metric_names = {m.name for m in metrics}
        if self.scoring in metric_names:
            return self.scoring

        # 常见映射
        _SCORING_MAP = {
            "roc_auc": "auc",
            "accuracy": "accuracy",
        }
        mapped = _SCORING_MAP.get(self.scoring)
        if mapped and mapped in metric_names:
            return mapped

        # 兜底：直接用 scoring 名
        return self.scoring

    def _check_is_fitted(self):
        """检查是否已完成实验。"""
        if not hasattr(self, "best_estimator_"):
            raise RuntimeError(
                "ExperimentRunner 尚未拟合，请先调用 fit() 方法"
            )
