"""
Optuna 自动调参器 — OptunaTuner

基于 Optuna 的贝叶斯超参搜索，专为风控分类器设计：
- 内置风控场景推荐搜索空间（RiskXGBClassifier 默认参数邻域）
- 支持 AUC / KS / F1 等风控常用评估指标
- 支持 early stopping 避免过拟合
- 搜索完成后自动用最优参数重新训练完整模型
- 兼容任意 sklearn 分类器，不限于 XGBoost
"""

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_score
from sklearn.metrics import get_scorer

import optuna
from optuna.samplers import TPESampler


# 风控场景推荐搜索空间
_DEFAULT_SEARCH_SPACE = {
    "n_estimators": (100, 500),
    "max_depth": (3, 7),
    "learning_rate": (0.01, 0.3),
    "min_child_weight": (1, 10),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "reg_alpha": (0.0, 1.0),
    "reg_lambda": (0.0, 5.0),
    "scale_pos_weight": (1, 20),
}


class OptunaTuner(BaseEstimator):
    """
    基于 Optuna 的风控模型自动调参器。

    使用 TPE 采样器进行贝叶斯搜索，找到最优超参数后
    自动在全部训练数据上重新拟合。

    Parameters
    ----------
    estimator : sklearn 兼容分类器
        待调参的估计器实例，如 ``RiskXGBClassifier()``。
    n_trials : int, default=50
        Optuna 搜索试验次数。
    search_space : dict or None, default=None
        自定义搜索空间。格式为 ``{参数名: (low, high)}``，
        数值参数自动识别 int/float。为 None 时使用风控默认搜索空间。
    scoring : str, default="roc_auc"
        评估指标名称，支持 sklearn 可调用指标和风控常用指标：
        "roc_auc" / "f1" / "ks" / "accuracy" 等。
    cv : int, default=5
        交叉验证折数。
    n_jobs : int, default=1
        并行任务数（传递给 cross_val_score）。
    random_state : int or None, default=42
        随机种子（同时用于 Optuna sampler 和交叉验证）。
    early_stopping_rounds : int or None, default=10
        Optuna 剪枝的提前停止轮数（MedianPruner）。
        为 None 时不启用剪枝。
    verbose : int, default=0
        Optuna 日志级别。0=静默，1=进度条，2=详细。

    Example
    -------
    >>> from risk_ml import RiskXGBClassifier, OptunaTuner
    >>> tuner = OptunaTuner(
    ...     estimator=RiskXGBClassifier(),
    ...     n_trials=30,
    ...     scoring="roc_auc",
    ... )
    >>> tuner.fit(X_train, y_train)
    >>> print(tuner.best_params_)
    >>> y_score = tuner.predict_score(X_test)
    """

    def __init__(
        self,
        estimator,
        n_trials=50,
        search_space=None,
        scoring="roc_auc",
        cv=5,
        n_jobs=1,
        random_state=42,
        early_stopping_rounds=10,
        verbose=0,
    ):
        self.estimator = estimator
        self.n_trials = n_trials
        self.search_space = search_space
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose

    @staticmethod
    def _ks_scorer(estimator, X, y):
        """风控 KS 统计量评分函数。"""
        y_score = estimator.predict_proba(X)[:, 1]
        pos_score = y_score[y == 1]
        neg_score = y_score[y == 0]
        if len(pos_score) == 0 or len(neg_score) == 0:
            return 0.0
        # 计算 KS
        bins = np.linspace(0, 1, 101)
        pos_hist, _ = np.histogram(pos_score, bins=bins, density=True)
        neg_hist, _ = np.histogram(neg_score, bins=bins, density=True)
        pos_cdf = np.cumsum(pos_hist) / pos_hist.sum()
        neg_cdf = np.cumsum(neg_hist) / neg_hist.sum()
        ks = np.max(np.abs(pos_cdf - neg_cdf))
        return ks

    def _get_scorer(self):
        """获取评分函数，支持风控 KS 指标。"""
        if self.scoring == "ks":
            return self._ks_scorer
        try:
            return get_scorer(self.scoring)
        except ValueError:
            raise ValueError(
                f"不支持的评估指标: {self.scoring}，"
                f"可用指标见 sklearn.metrics.get_scorer_names() 或 'ks'"
            )

    def _suggest_params(self, trial, search_space):
        """
        根据搜索空间为一次试验采样参数。

        自动推断参数类型：整数值范围用 suggest_int，浮点值用 suggest_float。
        支持 sklearn Pipeline 的 step__param 嵌套参数格式。
        """
        params = {}
        for name, bounds in search_space.items():
            low, high = bounds
            # 判断参数类型：默认参数值在 estimator 中为 int 则用 int 采样
            default_val = self._get_param_default(name)
            if isinstance(default_val, int) and isinstance(low, int) and isinstance(high, int):
                params[name] = trial.suggest_int(name, low, high)
            else:
                # 对于 learning_rate 等可能跨度很大的参数，使用 log 采样
                use_log = low > 0 and high / low > 100
                params[name] = trial.suggest_float(name, low, high, log=use_log)
        return params

    def _get_param_default(self, name: str):
        """
        获取估计器参数的默认值，支持 Pipeline 的 step__param 格式。

        如 "classifier__n_estimators" 会查找 pipeline.named_steps.classifier.n_estimators。
        """
        val = getattr(self.estimator, name, None)
        if val is not None:
            return val

        # Pipeline step__param 格式
        if "__" in name and hasattr(self.estimator, "named_steps"):
            parts = name.split("__", 1)
            step_name, param_name = parts
            step = self.estimator.named_steps.get(step_name)
            if step is not None:
                return getattr(step, param_name, None)

        return None

    def fit(self, X, y, X_val=None, y_val=None, **fit_params):
        """
        执行超参搜索并拟合最优模型。

        支持两种评估模式：
        - CV 模式（X_val=None）：使用 cross_val_score 交叉验证评估
        - Holdout 模式（X_val/y_val 传入）：训练集 fit，验证集评分

        Parameters
        ----------
        X : pandas DataFrame 或 array-like
            训练特征。
        y : array-like
            目标变量（0/1 二分类）。
        X_val : pandas DataFrame 或 array-like, default=None
            验证集特征。传入后使用 holdout 评估替代 CV。
        y_val : array-like, default=None
            验证集标签。与 X_val 配合使用。
        **fit_params : dict
            传递给估计器 fit() 的额外参数（如 sample_weight）。
            对于 sklearn Pipeline，使用 ``{step_name}__sample_weight`` 格式。

        Returns
        -------
        self
        """
        search_space = self.search_space or _DEFAULT_SEARCH_SPACE
        scorer = self._get_scorer()
        sampler = TPESampler(seed=self.random_state)

        # 配置剪枝器
        pruner = None
        if self.early_stopping_rounds is not None:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=0,
                interval_steps=1,
            )

        optuna.logging.set_verbosity(
            optuna.logging.DEBUG if self.verbose >= 2
            else optuna.logging.INFO if self.verbose >= 1
            else optuna.logging.WARNING
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        # --- Holdout 模式：验证集直接评估 ---
        if X_val is not None and y_val is not None:
            # 记录评估模式
            self._eval_mode = "holdout"

            def _objective(trial):
                params = self._suggest_params(trial, search_space)
                est = clone(self.estimator).set_params(**params)
                # 训练集 fit
                est.fit(X, y, **fit_params)
                # 验证集评分
                return scorer(est, X_val, y_val)

            study.optimize(
                _objective,
                n_trials=self.n_trials,
                show_progress_bar=self.verbose >= 1,
            )

            # 记录搜索结果
            self.best_params_ = study.best_params
            self.best_score_ = study.best_value
            self.study_ = study
            self.trials_dataframe_ = study.trials_dataframe()

            # 用最优参数在全部训练数据上重新训练
            self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
            self.best_estimator_.fit(X, y, **fit_params)

        # --- CV 模式：交叉验证评估（原逻辑） ---
        else:
            # 记录评估模式
            self._eval_mode = "cv"

            cv_fit_params = fit_params

            def _objective(trial):
                params = self._suggest_params(trial, search_space)
                est = clone(self.estimator).set_params(**params)
                scores = cross_val_score(
                    est, X, y,
                    scoring=scorer,
                    cv=self.cv,
                    n_jobs=self.n_jobs,
                    params=cv_fit_params,
                )
                return scores.mean()

            study.optimize(
                _objective,
                n_trials=self.n_trials,
                show_progress_bar=self.verbose >= 1,
            )

            # 记录搜索结果
            self.best_params_ = study.best_params
            self.best_score_ = study.best_value
            self.study_ = study
            self.trials_dataframe_ = study.trials_dataframe()

            # 用最优参数在全部数据上重新训练
            self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
            self.best_estimator_.fit(X, y, **fit_params)

        return self

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

    def _check_is_fitted(self):
        """检查是否已完成调参和拟合。"""
        if not hasattr(self, "best_estimator_"):
            raise RuntimeError(
                "OptunaTuner 尚未拟合，请先调用 fit() 方法"
            )

    def _more_tags(self):
        """sklearn 标签。"""
        return {"binary_only": True}
