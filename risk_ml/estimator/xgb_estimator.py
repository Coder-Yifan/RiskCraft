"""
风控 XGBoost 分类器 — RiskXGBClassifier

对 xgboost.XGBClassifier 的风控场景封装：
- 默认参数面向信贷风控场景调优（scale_pos_weight、max_depth 等）
- 自动记录训练时的特征名
- 提供 predict_score() 方法直接返回正例概率
- 完全兼容 sklearn 接口（fit / predict / predict_proba / get_params / set_params）
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier


class RiskXGBClassifier(BaseEstimator, ClassifierMixin):
    """
    风控场景 XGBoost 二分类器。

    在 XGBClassifier 基础上提供风控默认参数和便捷接口，
    完全兼容 sklearn 生态（Pipeline、GridSearchCV 等）。

    Parameters
    ----------
    n_estimators : int, default=200
        提升轮数。
    max_depth : int, default=4
        树最大深度。风控场景建议 3~5，避免过拟合。
    learning_rate : float, default=0.05
        学习率。风控场景常用 0.01~0.1。
    scale_pos_weight : float, default=1
        正负样本权重比。风控场景正例极少，可设为
        ``neg_count / pos_count`` 以平衡样本。
    min_child_weight : float, default=5
        子节点最小权重和。风控场景建议偏大，增强泛化。
    subsample : float, default=0.8
        行采样比例。
    colsample_bytree : float, default=0.8
        列采样比例。
    reg_alpha : float, default=0.1
        L1 正则化系数。
    reg_lambda : float, default=1.0
        L2 正则化系数。
    random_state : int or None, default=42
        随机种子。
    eval_metric : str, default="auc"
        评估指标。
    tree_method : str, default="hist"
        树构建算法。
    n_jobs : int, default=-1
        并行线程数。
    **xgb_kwargs : dict
        传递给 XGBClassifier 的其他参数。

    Example
    -------
    >>> clf = RiskXGBClassifier(scale_pos_weight=10)
    >>> clf.fit(X_train, y_train)
    >>> y_score = clf.predict_score(X_test)
    """

    def __init__(
        self,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=1,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        **xgb_kwargs,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.eval_metric = eval_metric
        self.tree_method = tree_method
        self.n_jobs = n_jobs
        self.xgb_kwargs = xgb_kwargs

    def _build_xgb(self):
        """根据当前参数构建 XGBClassifier 实例。"""
        params = self.get_params()
        # 移除非 XGBClassifier 原生参数
        params.pop("xgb_kwargs", None)
        params.update(self.xgb_kwargs)
        return XGBClassifier(**params)

    def fit(self, X, y, **fit_kwargs):
        """
        拟合 XGBoost 分类器。

        当输入为 pandas DataFrame 时，自动将 object 类型列转换为
        category 类型，以兼容 XGBoost 原生分类特征支持。

        Parameters
        ----------
        X : pandas DataFrame 或 array-like
            训练特征。
        y : array-like
            目标变量（0/1 二分类）。
        **fit_kwargs : dict
            传递给 XGBClassifier.fit() 的额外参数，
            如 eval_set、early_stopping_rounds 等。

        Returns
        -------
        self
        """
        # 记录特征名
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            self.n_features_in_ = X.shape[1]
            # 自动将 object 列转为 category，启用 XGBoost 原生分类支持
            cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
            if cat_cols:
                X = X.copy()
                for col in cat_cols:
                    X[col] = X[col].astype("category")
                self._has_categorical_ = True
            else:
                self._has_categorical_ = False

        self.model_ = self._build_xgb()
        if getattr(self, "_has_categorical_", False):
            self.model_.set_params(enable_categorical=True)
        self.model_.fit(X, y, **fit_kwargs)

        # 暴露 classes_ 属性以兼容 sklearn
        self.classes_ = self.model_.classes_
        return self

    def _prepare_input(self, X):
        """预测前预处理输入（与 fit 保持一致的类别列转换）。"""
        if isinstance(X, pd.DataFrame) and getattr(self, "_has_categorical_", False):
            X = X.copy()
            for col in X.select_dtypes(include=["object"]).columns:
                X[col] = X[col].astype("category")
        return X

    def predict(self, X):
        """
        预测类别标签。

        Parameters
        ----------
        X : pandas DataFrame 或 array-like
            输入特征。

        Returns
        -------
        np.ndarray
            预测的类别标签。
        """
        check_is_fitted(self, "model_")
        return self.model_.predict(self._prepare_input(X))

    def predict_proba(self, X):
        """
        预测类别概率。

        Parameters
        ----------
        X : pandas DataFrame 或 array-like
            输入特征。

        Returns
        -------
        np.ndarray
            形状 (n_samples, 2) 的概率矩阵。
        """
        check_is_fitted(self, "model_")
        return self.model_.predict_proba(self._prepare_input(X))

    def predict_score(self, X):
        """
        预测正例概率（风控评分）。

        等价于 ``predict_proba(X)[:, 1]``，
        返回每个样本为正例的概率。

        Parameters
        ----------
        X : pandas DataFrame 或 array-like
            输入特征。

        Returns
        -------
        np.ndarray
            形状 (n_samples,) 的正例概率数组。
        """
        return self.predict_proba(X)[:, 1]

    def feature_importance(self, importance_type="gain"):
        """
        获取特征重要性。

        Parameters
        ----------
        importance_type : str, default="gain"
            重要性类型，可选 "weight"/"gain"/"cover"/"total_gain"/"total_cover"。

        Returns
        -------
        dict
            特征名 → 重要性值的字典。
        """
        check_is_fitted(self, "model_")
        booster = self.model_.get_booster()
        raw = booster.get_score(importance_type=importance_type)
        if hasattr(self, "feature_names_in_"):
            # 补全未出现的特征（重要性为 0）
            return {f: raw.get(f, 0.0) for f in self.feature_names_in_}
        return raw

    def _more_tags(self):
        """sklearn 标签：二分类器。"""
        return {"binary_only": True}
