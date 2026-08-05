"""
executor 侧打分引擎（禁止 import risk_ml）

对照 risk_ml/online_deploy/_model_onnx.py / _model_m2cgen.py 抄写：
- OnnxEngine    : 懒建 onnxruntime session + float32 输入（对照 _model_onnx.py:124-135）
- M2CgenEngine  : 懒 exec + numpy 向量化 float32 量化（对照 _model_m2cgen.py:103-115）

onnxruntime 为惰性导入（仅 onnx 后端首次打分时加载），
m2cgen 后端零第三方依赖，为 Spark 集群默认推荐。
"""

import numpy as np


class OnnxEngine:
    """ONNX Runtime 打分引擎。"""

    def __init__(self, model_bytes, feature_names, base_score):
        self.model_bytes = bytes(model_bytes)
        self.feature_names = list(feature_names)
        self.base_score = float(base_score)
        self._session = None

    def score(self, X):
        """批量打分，X 形状 (n, f) → 正例概率 (n,)。"""
        import onnxruntime as ort

        if self._session is None:
            self._session = ort.InferenceSession(
                self.model_bytes, providers=["CPUExecutionProvider"]
            )
        out = self._session.run(
            None, {"input": np.asarray(X, dtype=np.float32)}
        )[0]
        return out[:, 0]


class M2CgenEngine:
    """float32 修正版纯 Python 打分引擎（零依赖）。"""

    def __init__(self, code, feature_names, base_score):
        self.code = code
        self.feature_names = list(feature_names)
        self.base_score = float(base_score)
        self._fn = None

    def score(self, X):
        """批量打分，X 形状 (n, f) → 正例概率 (n,)。

        输入 float32 量化在此向量化完成（astype float32→float64，
        与 struct._f32 舍入 bit 级一致），生成代码仅做树遍历。
        """
        if self._fn is None:
            ns = {}
            exec(compile(self.code, "<m2cgen>", "exec"), ns)
            self._fn = ns["score"]
        f = self._fn
        Xq = np.asarray(X, dtype=np.float64).astype(np.float32).astype(np.float64)
        return np.array([f(row.tolist()) for row in Xq])
