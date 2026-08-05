"""
风控 LightGBM 分类器 — RiskLGBMClassifier

对 lightgbm.LGBMClassifier 的风控场景封装，与 RiskXGBClassifier 对齐：
- 同一套风控默认参数（scale_pos_weight、learning_rate、正则等）
- 自动记录训练时的特征名
- predict_score() 直接返回正例概率
- to_deploy_model() 返回归一化 TreeModel（LGB 概率 = sigmoid(Σleaf)，
  base_margin = 0；数值分裂 decision_type "<=" → LE 模式）
- 完全兼容 sklearn 接口（fit / predict / predict_proba / get_params / set_params）
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.utils.validation import check_is_fitted

from .base_estimator import RiskEstimator


class RiskLGBMClassifier(RiskEstimator):
    """
    风控场景 LightGBM 二分类器。

    在 LGBMClassifier 基础上提供风控默认参数和便捷接口，
    完全兼容 sklearn 生态（Pipeline、GridSearchCV 等）。

    Parameters
    ----------
    n_estimators : int, default=200
        提升轮数。
    num_leaves : int, default=31
        叶子数上限。
    max_depth : int, default=-1
        树最大深度（-1 表示不限）。
    learning_rate : float, default=0.05
        学习率。风控场景常用 0.01~0.1。
    scale_pos_weight : float, default=1
        正负样本权重比。风控场景正例极少，可设为
        ``neg_count / pos_count`` 以平衡样本。
    min_child_samples : int, default=20
        叶子最小样本数。风控场景建议偏大，增强泛化。
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
    n_jobs : int, default=-1
        并行线程数。
    **lgb_kwargs : dict
        传递给 LGBMClassifier 的其他参数。

    Example
    -------
    >>> clf = RiskLGBMClassifier(scale_pos_weight=10)
    >>> clf.fit(X_train, y_train)
    >>> y_score = clf.predict_score(X_test)
    """

    def __init__(
        self,
        n_estimators=200,
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.05,
        scale_pos_weight=1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        **lgb_kwargs,
    ):
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.lgb_kwargs = lgb_kwargs

    def _build_lgb(self):
        """根据当前参数构建 LGBMClassifier 实例。"""
        params = self.get_params()
        params.pop("lgb_kwargs", None)
        params.update(self.lgb_kwargs)
        return LGBMClassifier(**params)

    def fit(self, X, y, **fit_kwargs):
        """
        拟合 LightGBM 分类器。

        当输入为 pandas DataFrame 时，自动将 object 类型列转换为
        category 类型，以兼容 LightGBM 原生分类特征支持。

        Parameters
        ----------
        X : pandas DataFrame 或 array-like
            训练特征。
        y : array-like
            目标变量（0/1 二分类）。
        **fit_kwargs : dict
            传递给 LGBMClassifier.fit() 的额外参数，
            如 eval_set、callbacks 等。

        Returns
        -------
        self
        """
        cat_cols = self._set_feature_meta(X)
        if cat_cols:
            X = X.copy()
            for col in cat_cols:
                X[col] = X[col].astype("category")

        self.model_ = self._build_lgb()
        if cat_cols:
            self.model_.set_params(categorical_feature=cat_cols)
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

    def to_deploy_model(self):
        """
        返回归一化 TreeModel（供在线部署双后端使用）。

        与 XGBoost 不同：LGB 概率 = sigmoid(Σleaf)，无 base 偏移
        （base_margin = 0）；阈值 float32 舍入、叶子 double 全精度。
        """
        from risk_ml.online_deploy._tree_model import TreeModel

        check_is_fitted(self, "model_")
        booster = self.model_.booster_
        feature_names = self._deploy_feature_names(booster.feature_name())
        return TreeModel.from_lgb_booster(booster, feature_names)

    def feature_importance(self, importance_type="gain"):
        """
        获取特征重要性。

        Parameters
        ----------
        importance_type : str, default="gain"
            重要性类型，可选 "split"/"gain"。

        Returns
        -------
        dict
            特征名 → 重要性值的字典。
        """
        check_is_fitted(self, "model_")
        booster = self.model_.booster_
        vals = np.asarray(booster.feature_importance(importance_type=importance_type))
        names = booster.feature_name()
        raw = dict(zip(names, vals.tolist()))
        if hasattr(self, "feature_names_in_"):
            # 补全未出现的特征（重要性为 0）
            return {f: raw.get(f, 0.0) for f in self.feature_names_in_}
        return raw
