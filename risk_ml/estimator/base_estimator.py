"""
估计器抽象基类 — RiskEstimator

定义估计器的统一在线部署契约 ``to_deploy_model()``：
- 子类实现 ``to_deploy_model()``，返回框架无关的 ``TreeModel``
  （归一化树结构，阈值/叶子/base_margin 已按框架量化，见 online_deploy._tree_model）
- 在线部署双后端（m2cgen / onnx）只认 TreeModel，与具体框架完全解耦

新增估计器只需：
1. 继承 RiskEstimator 并实现 fit / to_deploy_model
2. predict / predict_proba / predict_score 由基类复用（底层模型对象
   暴露同名 sklearn 接口即可）；feature_importance 按框架覆写
"""

from abc import ABC, abstractmethod

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class RiskEstimator(BaseEstimator, ClassifierMixin, ABC):
    """
    估计器抽象基类。

    Attributes
    ----------
    feature_names_in_ : list[str]   (fit 时由 DataFrame 列名记录)
    n_features_in_ : int
    _has_categorical_ : bool        (训练是否含分类特征列)
    model_ : 底层模型对象（XGBClassifier / LGBMClassifier）
    classes_ : np.ndarray
    """

    @abstractmethod
    def to_deploy_model(self):
        """
        返回归一化 TreeModel，供在线部署双后端（m2cgen / onnx）使用。

        需底层模型（self.model_）已拟合。实现方负责：
        - 从底层 booster 归一化（阈值/叶子/base_margin 按框架量化）
        - 特征顺序与 fit 时一致（feature_names_in_）

        Returns
        -------
        online_deploy._tree_model.TreeModel
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 共享辅助
    # ------------------------------------------------------------------
    def _set_feature_meta(self, X):
        """记录特征元信息，返回 object 列名列表（需调用方转 category）。"""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            self.n_features_in_ = X.shape[1]
            cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
            self._has_categorical_ = bool(cat_cols)
            return cat_cols
        self._has_categorical_ = False
        return []

    def _deploy_feature_names(self, booster_names):
        """部署特征名：优先 fit 记录的列名，回退底层 booster 名。"""
        return list(getattr(self, "feature_names_in_", None) or booster_names or [])

    # ------------------------------------------------------------------
    # 通用接口（子类按需覆写 _prepare_input）
    # ------------------------------------------------------------------
    def _prepare_input(self, X):
        """预测前预处理输入。子类按需覆写（如类别列转换）。"""
        return X

    def predict(self, X):
        """
        预测类别标签。

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

        等价于 ``predict_proba(X)[:, 1]``。

        Returns
        -------
        np.ndarray
            形状 (n_samples,) 的正例概率数组。
        """
        return self.predict_proba(X)[:, 1]

    def _more_tags(self):
        """sklearn 标签：二分类器。"""
        return {"binary_only": True}
