"""LabelDefinition 算子 — 1.3标签定义。"""

import pandas as pd

from .._base import ReportOperator, SubSection


class LabelDefinitionOperator(ReportOperator):
    """标签定义 — Y=0/好, Y=-1/灰, Y=1/坏 + 观察期。"""

    @property
    def name(self) -> str:
        return "label_definition"

    @property
    def title(self) -> str:
        return "1.3标签定义"

    def compute(self, context) -> list[SubSection]:
        rows = []
        for label_val, label_desc in context.label_definition.items():
            rows.append({"标签": f"Y={label_val}", "描述": label_desc})

        df = pd.DataFrame(rows)
        if context.observation_period:
            df_note = pd.DataFrame([{"项目": "观察期", "内容": context.observation_period}])
            return [SubSection(self.title, df), SubSection("观察期", df_note)]

        return [SubSection(self.title, df)]
