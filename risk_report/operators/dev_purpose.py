"""DevPurpose 算子 — 1.1模型开发目的。"""

import pandas as pd

from .._base import ReportOperator, SubSection


class DevPurposeOperator(ReportOperator):
    """模型开发目的 — 目标函数+使用场景表。"""

    @property
    def name(self) -> str:
        return "dev_purpose"

    @property
    def title(self) -> str:
        return "1.1模型开发目的"

    def compute(self, context) -> list[SubSection]:
        rows = [
            {"项目": "背景", "内容": context.background or "未填写"},
            {"项目": "应用场景", "内容": context.application or "未填写"},
            {"项目": "标签列", "内容": context.label_col or "未填写"},
            {"项目": "观察期", "内容": context.observation_period or "未填写"},
        ]
        return [SubSection(self.title, pd.DataFrame(rows))]
