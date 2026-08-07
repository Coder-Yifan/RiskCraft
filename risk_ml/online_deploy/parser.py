"""
PipelineParser — 将已拟合的 sklearn pipeline 编译为在线部署流水线

编译过程：
1. 遍历 pipeline 步骤，将每个 transformer 映射为纯 numpy 部署算子
2. 将最终估计器（RiskEstimator → to_deploy_model）映射为树模型后端（onnx / m2cgen，xgb / lgb 共用）
3. 记录输入列，供 score(dict) / score_batch 使用

序列化：
- to_dict / from_dict：JSON 友好 dict（无 pickle，无 numpy 类型）
- to_json / from_json：字符串
"""

import json

import numpy as np

from ._base import DeployOp, json_dumps, json_safe
from ._model_m2cgen import M2CgenBackend
from ._model_onnx import OnnxBackend
from ._ops import BinOp, BinWoeOp, CleanerOp, DeriveOp, SelectOp, WoeOp
from .exceptions import DeployError, UnsupportedStepError
from .registry import build_deploy_op

# 算子类型 → 反序列化类
_OP_CLASSES = {
    CleanerOp.kind: CleanerOp,
    BinOp.kind: BinOp,
    WoeOp.kind: WoeOp,
    BinWoeOp.kind: BinWoeOp,
    SelectOp.kind: SelectOp,
    DeriveOp.kind: DeriveOp,
}
# 模型后端类型 → 反序列化类
_MODEL_CLASSES = {
    "onnx": OnnxBackend,
    "m2cgen": M2CgenBackend,
}


class DeployPipeline:
    """编译后的在线部署流水线。

    Example
    -------
    >>> deploy = PipelineParser(backend="onnx").compile_pipeline(pipe)
    >>> deploy.score({"amount": 3000, "age": 35})            # 单条
    >>> deploy.score_batch([{"amount": 3000}, {...}])        # 批量
    >>> deploy.to_json()                                     # 序列化
    """

    def __init__(self, ops, model_op, feature_names_in):
        self.ops = list(ops)                 # list[DeployOp]
        self.model_op = model_op             # OnnxBackend / M2CgenBackend
        self.feature_names_in_ = list(feature_names_in)

    # ------------------------------------------------------------------
    # 打分
    # ------------------------------------------------------------------
    def _to_array(self, rows):
        """list[dict] → numpy 数组 (n, f)，缺失/非数值 → NaN。"""
        n = len(rows)
        arr = np.full((n, len(self.feature_names_in_)), np.nan)
        for r_i, row in enumerate(rows):
            for i, c in enumerate(self.feature_names_in_):
                v = row.get(c)
                if v is None:
                    continue
                try:
                    arr[r_i, i] = v
                except (TypeError, ValueError):
                    pass  # 非数值值 → NaN
        return arr

    def score(self, row):
        """单条打分：dict → 正例概率。"""
        X = self._to_array([row])
        for op in self.ops:
            X = op.transform(X)
        return float(self.model_op.score(X)[0])

    def score_batch(self, rows):
        """批量打分：list[dict] → 正例概率数组 (n,)。"""
        if not rows:
            return np.array([])
        X = self._to_array(rows)
        for op in self.ops:
            X = op.transform(X)
        return self.model_op.score(X)

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def to_dict(self):
        return {
            "version": 1,
            "feature_names_in_": self.feature_names_in_,
            "ops": [op.to_dict() for op in self.ops],
            "model": self.model_op.to_dict(),
        }

    def to_json(self):
        return json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d):
        if d.get("version") != 1:
            raise DeployError(f"不支持的部署格式版本: {d.get('version')}")
        ops = []
        for op_d in d["ops"]:
            op_cls = _OP_CLASSES.get(op_d["kind"])
            if op_cls is None:
                raise DeployError(f"未知算子类型: {op_d['kind']}")
            ops.append(op_cls.from_dict(op_d))
        model_d = d["model"]
        model_cls = _MODEL_CLASSES.get(model_d["kind"])
        if model_cls is None:
            raise DeployError(f"未知模型后端: {model_d['kind']}")
        return cls(ops, model_cls.from_dict(model_d), d["feature_names_in_"])

    @classmethod
    def from_json(cls, s):
        return cls.from_dict(json.loads(s))

    # ------------------------------------------------------------------
    # 展示
    # ------------------------------------------------------------------
    def describe(self):
        lines = [f"DeployPipeline 输入 {len(self.feature_names_in_)} 特征:"]
        for op in self.ops:
            lines.append(f"  {op.describe()}")
        lines.append(f"  {self.model_op.describe()}")
        return "\n".join(lines)

    def __repr__(self):
        return self.describe()


class PipelineParser:
    """将已拟合 pipeline 编译为 DeployPipeline。

    Parameters
    ----------
    backend : str, default="m2cgen"
        树模型后端（xgb / lgb 共用 TreeModel），可选：
        - "m2cgen": 纯 Python 转译，单条 ~10us，零依赖
        - "onnx":   ONNX Runtime，单条 ~16us，跨语言可复用
    """

    def __init__(self, backend="m2cgen"):
        if backend not in ("m2cgen", "onnx"):
            raise ValueError(f"不支持的 backend: {backend!r}，可选 'm2cgen' / 'onnx'")
        self.backend = backend

    # ------------------------------------------------------------------
    # 编译
    # ------------------------------------------------------------------
    def compile_pipeline(self, pipe):
        """编译已拟合的 sklearn Pipeline / RiskPipeline。"""
        steps = getattr(pipe, "steps", None)
        if not steps:
            raise DeployError("输入必须是已拟合的 sklearn Pipeline（含 steps 属性）")

        ops = []
        columns = None
        input_columns = None
        model_op = None

        for idx, (name, step) in enumerate(steps):
            if step is None or step == "passthrough":
                continue
            if columns is None:
                columns = list(getattr(step, "feature_names_in_", []))
                input_columns = list(columns)  # 记录原始输入列（模型链路入口）

            is_last = idx == len(steps) - 1
            # 最终估计器：具备 predict_proba 的步骤
            if is_last and hasattr(step, "predict_proba"):
                # 评分拉伸折叠进模型 margin（score_scaler 由 pipeline 持有）
                scaler = getattr(pipe, "score_scaler", None)
                model_op = self._build_model(step, columns, score_scaler=scaler)
                continue

            op = self._build_op(step, columns, name)
            if op is None:
                raise UnsupportedStepError(
                    f"步骤 '{name}'（{type(step).__name__}）不支持在线部署："
                    "请通过 register_deploy_builder 注册自定义构建器，"
                    "或使用支持的内置算子"
                )
            ops.append(op)
            columns = op.output_columns

        if model_op is None:
            raise DeployError("pipeline 末尾缺少具备 predict_proba 的估计器")

        return DeployPipeline(ops, model_op, input_columns)

    # ------------------------------------------------------------------
    # 单步构建
    # ------------------------------------------------------------------
    def _build_op(self, step, columns, name):
        """内置算子识别 + 自定义注册表。"""
        # 自定义算子优先（注册表 / to_deploy 协议）
        op = build_deploy_op(step, columns)
        if op is not None:
            if not isinstance(op, DeployOp):
                raise DeployError(
                    f"步骤 '{name}' 的 to_deploy 返回类型必须为 DeployOp，"
                    f"收到 {type(op).__name__}"
                )
            return op

        # 内置算子（按继承优先级从子类到基类）
        from ..binning import ChiMergeBinner
        from ..binning.base_binner import BaseBinner
        from ..encoding import BaseEncoder
        from ..preprocessing import FeatureCleaner, FeatureDerivativeTransformer
        from .._base import RiskSelector

        if isinstance(step, FeatureCleaner):
            return CleanerOp.from_step(step, columns, name)
        if isinstance(step, FeatureDerivativeTransformer):
            return DeriveOp.from_step(step, columns, name)
        if isinstance(step, ChiMergeBinner):
            return BinOp.from_step(step, columns, name)
        if isinstance(step, BaseBinner):
            return BinOp.from_step(step, columns, name)
        if isinstance(step, BaseEncoder):
            if hasattr(step, "binner_"):  # 内嵌分箱 → 分箱+编码联合算子
                return BinWoeOp.from_step(step, columns, name)
            return WoeOp.from_step(step, columns, name)
        if isinstance(step, RiskSelector):
            return SelectOp.from_step(step, columns, name)
        return None

    def _build_model(self, step, columns, score_scaler=None):
        """将 RiskEstimator 编译为树模型后端（m2cgen / onnx）。

        score_scaler: ScoreScaler or None，评分拉伸算子。配置后折叠进 margin
        （score = offset + scale·margin），后端输出风险分而非概率。
        """
        if getattr(step, "_has_categorical_", False):
            raise UnsupportedStepError(
                "模型训练时包含分类特征（category 列），ONNX/m2cgen 后端"
                "仅支持数值特征，请先对分类特征做数值编码"
            )

        to_deploy = getattr(step, "to_deploy_model", None)
        if to_deploy is None:
            raise UnsupportedStepError(
                f"最终估计器 {type(step).__name__} 不支持部署："
                "仅支持继承 risk_ml.estimator.RiskEstimator 的估计器"
            )
        tree_model = to_deploy()

        feature_names = tree_model.feature_names
        if set(feature_names) != set(columns):
            raise UnsupportedStepError(
                f"模型输入列 {feature_names} 与上游输出列 {columns} 不一致"
            )

        if self.backend == "onnx":
            return OnnxBackend.from_tree_model(tree_model, score_scaler=score_scaler)
        return M2CgenBackend.from_tree_model(tree_model, score_scaler=score_scaler)
