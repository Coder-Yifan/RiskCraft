"""ModelAssumption 算子 — 1.2模型假设。"""

import pandas as pd

from .._base import ReportOperator, SubSection, placeholder_df


class ModelAssumptionOperator(ReportOperator):
    """模型假设 — 展示模型参数与假设。"""

    @property
    def name(self) -> str:
        return "model_assumption"

    @property
    def title(self) -> str:
        return "1.2模型假设"

    def compute(self, context) -> list[SubSection]:
        attrs = context.pipeline_attrs
        if attrs is None or attrs.model_params_ is None:
            return [SubSection(self.title, placeholder_df(
                "流水线属性未提取，请通过 ReportContext.pipeline 传入已拟合流水线"
            ))]

        # 参数名中文映射
        desc_map = {
            "max_depth": "最大树深",
            "n_estimators": "决策树个数",
            "learning_rate": "学习速率",
            "eta": "学习速率",
            "subsample": "子树样本采样比例",
            "colsample_bytree": "子树特征采样比例",
            "reg_alpha": "L1正则化系数",
            "alpha": "L1正则化系数",
            "reg_lambda": "L2正则化系数",
            "lambda": "L2正则化系数",
            "min_child_weight": "叶子最小权重值",
        }

        rows = []
        for k, v in attrs.model_params_.items():
            rows.append({
                "参数名称解释": desc_map.get(k, k),
                "参数名称": k,
                "数值": str(v),
            })

        return [SubSection(self.title, pd.DataFrame(rows))]
