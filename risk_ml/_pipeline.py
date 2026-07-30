"""
风控建模流水线 — RiskPipeline

扩展 sklearn Pipeline，支持：
1. 验证集数据流：fit(X, y, X_val=None, y_val=None)
2. step间属性自动传递（如 IVSelector.iv_values_ → CorrelationSelector.iv_values）
3. PSISelector 在有验证集时正确计算 PSI（而非 PSI≈0）

不传 X_val 时，属性传递仍然生效，其余行为与 sklearn Pipeline 一致。
"""

from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.utils._user_interface import _print_elapsed_time


def _route_attributes(steps, fitted_step_idx):
    """
    将已拟合 step 的属性传递给后续 step 的 __init__ 参数。

    当前规则：
    - iv_values_ (pd.Series, 来自 IVSelector) → CorrelationSelector.iv_values
    - iv_values_ (dict, 来自 BinnerWoeEncoder) → CorrelationSelector.iv_values

    后续可根据需求扩展更多属性传递规则。
    """
    _, fitted_step = steps[fitted_step_idx]

    # 收集已拟合 step 的 IV 属性
    iv_val = None
    if hasattr(fitted_step, "iv_values_"):
        iv_val = fitted_step.iv_values_
    elif hasattr(fitted_step, "iv_values"):
        iv_val = getattr(fitted_step, "iv_values", None)

    if iv_val is None:
        return

    # 扫描后续 step，将 IV 注入 CorrelationSelector
    for idx in range(fitted_step_idx + 1, len(steps)):
        _, step = steps[idx]
        if hasattr(step, "iv_values") and step.iv_values is None:
            step.set_params(iv_values=iv_val)


class RiskPipeline(Pipeline):
    """
    控建模流水线：扩展 sklearn Pipeline，支持验证集和属性传递。

    核心扩展：
    - ``fit(X, y, X_val=None, y_val=None, **fit_params)`` — 传入验证集时，
      每个 transformer step 同时维护训练集和验证集的数据流，
      PSISelector 先 transform(X_val) 计算真实 PSI 再 transform(X_train)。
    - step间属性自动传递：fit完 IVSelector/BinnerWoeEncoder 后，
      自动将 iv_values_ 注入后续 CorrelationSelector(iv_values=None)。
    - 最终估计器若为 OptunaTuner 且传入 X_val/y_val，
      自动切换为 holdout 评估模式。

    不传 X_val 时，属性传递仍然生效，其余行为与 sklearn Pipeline 一致。

    Parameters
    ----------
    steps : list[tuple[str, estimator]]
        同 sklearn Pipeline，格式 [(name, estimator), ...]
    memory : str or object with joblib.Memory interface, default=None
        同 sklearn Pipeline
    verbose : bool, default=False
        同 sklearn Pipeline

    Attributes
    ----------
    X_val_transformed_ : pd.DataFrame or None
        fit 后存储经过所有 transformer step 变换后的验证集特征。
        仅在 fit 时传入 X_val 时存在。
    y_val_ : array-like or None
        fit 后存储验证集标签。仅在 fit 时传入 y_val 时存在。

    Example
    -------
    >>> from risk_ml import RiskPipeline, FeatureCleaner, BinnerWoeEncoder
    >>> pipe = RiskPipeline([
    ...     ("cleaner", FeatureCleaner()),
    ...     ("binner_woe", BinnerWoeEncoder()),
    ...     ("iv_selector", IVSelector()),
    ...     ("corr_selector", CorrelationSelector()),
    ...     ("classifier", RiskXGBClassifier()),
    ... ])
    >>> # 无验证集 — 属性传递生效，其余行为同 sklearn Pipeline
    >>> pipe.fit(X_train, y_train)
    >>> # 有验证集 — PSI 正确计算，OptunaTuner holdout 评估
    >>> pipe.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    """

    def fit(self, X, y=None, X_val=None, y_val=None, **fit_params):
        """
        拟合流水线，支持验证集数据流和属性传递。

        Parameters
        ----------
        X : pd.DataFrame
            训练集特征。
        y : array-like or None
            训练集标签。
        X_val : pd.DataFrame or None
            验证集特征。不传时不维护验证集数据流。
        y_val : array-like or None
            验证集标签。与 X_val 配合使用。
        **fit_params : dict
            传递给各 step fit 方法的额外参数，格式 ``step__param=value``。

        Returns
        -------
        self
        """
        routed_params = self._check_method_params(method="fit", props=fit_params)

        self.steps = list(self.steps)
        self._validate_steps()

        Xt = X   # 训练集逐步变换
        Xv = X_val  # 验证集逐步变换（None 时跳过）

        # ---- 1. Transformer steps ----
        for step_idx, name, transformer in self._iter(
            with_final=False, filter_passthrough=False
        ):
            if transformer is None or transformer == "passthrough":
                continue

            cloned = clone(transformer)

            # 在训练集上 fit
            step_params = routed_params.get(name, {})
            fit_kw = step_params.get("fit", {})
            with _print_elapsed_time("Pipeline", self._log_message(step_idx)):
                cloned.fit(Xt, y, **fit_kw)

            # 替换为已拟合的 step
            self.steps[step_idx] = (name, cloned)

            # step间属性传递（如 iv_values_ → CorrelationSelector.iv_values）
            _route_attributes(self.steps, step_idx)

            # PSISelector 特殊处理：先 transform(X_val) 计算真实 PSI，
            # 再用相同 mask 过滤 X_train（避免 transform(X_train) 覆盖 psi_values_）
            # 识别方式：fit 后有 reference_dist_ 属性（PSISelector 独有）
            if Xv is not None and hasattr(cloned, "reference_dist_"):
                Xv = cloned.transform(Xv)  # 计算真实 PSI，筛选 X_val
                # 用 _get_support_mask() 对 X_train 施加相同筛选
                mask = cloned._get_support_mask()
                keep_cols = [
                    c for c, m in zip(cloned.feature_names_in_, mask) if m
                ]
                Xt = Xt[keep_cols]
            else:
                Xt = cloned.transform(Xt)
                if Xv is not None:
                    Xv = cloned.transform(Xv)

        # ---- 2. 最终估计器 ----
        last_step_name, last_step = self.steps[-1]
        last_step_params = routed_params.get(last_step_name, {})
        fit_kw_last = last_step_params.get("fit", {})

        if last_step != "passthrough":
            # 检测最终估计器是否为 OptunaTuner（支持 X_val/y_val holdout）
            # 使用 duck-typing：检查类名避免循环导入
            is_optuna_tuner = type(last_step).__name__ == "OptunaTuner"
            if is_optuna_tuner and Xv is not None and y_val is not None:
                with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
                    last_step.fit(Xt, y, X_val=Xv, y_val=y_val, **fit_kw_last)
            else:
                with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
                    last_step.fit(Xt, y, **fit_kw_last)

        # ---- 3. 存储变换后的验证集 ----
        if Xv is not None:
            self.X_val_transformed_ = Xv
            self.y_val_ = y_val
        else:
            # 清除可能残留的验证集属性
            if hasattr(self, "X_val_transformed_"):
                del self.X_val_transformed_
            if hasattr(self, "y_val_"):
                del self.y_val_

        return self
