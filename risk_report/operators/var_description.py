"""VarDescription 算子 — 1.变量描述。"""

import pandas as pd

from .._base import ReportOperator, SubSection


class VarDescriptionOperator(ReportOperator):
    """变量描述 — 说明文字。"""

    @property
    def name(self) -> str:
        return "var_description"

    @property
    def title(self) -> str:
        return "1.变量描述"

    def compute(self, context) -> list[SubSection]:
        attrs = context.pipeline_attrs
        n_features = attrs.n_features_in_ if attrs and attrs.n_features_in_ else "未知"
        feature_list = attrs.feature_names_in_ if attrs and attrs.feature_names_in_ else []

        desc = f"经过变量初筛后的特征共 {n_features} 个"
        if feature_list:
            desc += f"，入模特征列表: {', '.join(feature_list[:10])}"
            if len(feature_list) > 10:
                desc += f" 等 {len(feature_list)} 个"

        rows = [{"说明": desc, "备注": "详见附件3-变量描述"}]
        return [SubSection(self.title, pd.DataFrame(rows))]
