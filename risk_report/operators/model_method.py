"""ModelMethod 算子 — 1.建模方法选择。"""

import pandas as pd

from .._base import ReportOperator, SubSection, placeholder_df


class ModelMethodOperator(ReportOperator):
    """建模方法选择 — XGB/LGBM参数表 或 LR系数表。"""

    @property
    def name(self) -> str:
        return "model_method"

    @property
    def title(self) -> str:
        return "1.建模方法选择"

    def compute(self, context) -> list[SubSection]:
        attrs = context.pipeline_attrs
        if attrs is None or attrs.model_params_ is None:
            return [SubSection(self.title, placeholder_df(
                "模型参数未提取，请通过 ReportContext.pipeline 传入"
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

        # 优先展示关键参数
        key_params = [
            "max_depth", "n_estimators", "learning_rate", "eta",
            "subsample", "colsample_bytree",
            "reg_alpha", "alpha", "reg_lambda", "lambda",
            "min_child_weight",
        ]

        rows = []
        for name in key_params:
            val = attrs.model_params_.get(name)
            if val is not None:
                rows.append({
                    "参数名称解释": desc_map.get(name, name),
                    "参数名称": name,
                    "数值": str(val),
                })

        # 补充非关键参数
        for k, v in attrs.model_params_.items():
            if k not in key_params and k not in {"steps", "memory", "verbose"}:
                rows.append({
                    "参数名称解释": desc_map.get(k, k),
                    "参数名称": k,
                    "数值": str(v)[:30],  # 截断过长值
                })

        return [SubSection(self.title, pd.DataFrame(rows))]
