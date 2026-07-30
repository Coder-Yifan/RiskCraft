"""VarCleaning 算子 — 2.变量清洗。"""

import pandas as pd

from .._base import ReportOperator, SubSection, placeholder_df


class VarCleaningOperator(ReportOperator):
    """变量清洗 — 清洗步骤表（缺失值/异常值/低方差）。"""

    @property
    def name(self) -> str:
        return "var_cleaning"

    @property
    def title(self) -> str:
        return "2.变量清洗"

    def compute(self, context) -> list[SubSection]:
        attrs = context.pipeline_attrs
        if attrs is None:
            return [SubSection(self.title, placeholder_df(
                "流水线属性未提取，请通过 ReportContext.pipeline 传入"
            ))]

        rows = []

        # 删除列
        if attrs.drop_columns_:
            cols_str = ", ".join(attrs.drop_columns_[:10])
            if len(attrs.drop_columns_) > 10:
                cols_str += f" 等{len(attrs.drop_columns_)}个"
            rows.append({
                "清洗步骤": "删除列",
                "数量": len(attrs.drop_columns_),
                "列名": cols_str,
            })

        # 缺失填充
        if attrs.impute_values_:
            rows.append({
                "清洗步骤": "缺失填充",
                "数量": len(attrs.impute_values_),
                "列名": "各列独立填充值",
            })

        # 缺失率阈值
        if attrs.missing_threshold_ is not None:
            rows.append({
                "清洗步骤": "缺失率阈值",
                "数量": "",
                "列名": f"阈值={attrs.missing_threshold_}",
            })

        # 低方差阈值
        if attrs.variance_threshold_ is not None:
            rows.append({
                "清洗步骤": "低方差阈值",
                "数量": "",
                "列名": f"阈值={attrs.variance_threshold_}",
            })

        if not rows:
            return [SubSection(self.title, placeholder_df("变量清洗逻辑需手动补充"))]

        return [SubSection(self.title, pd.DataFrame(rows))]
