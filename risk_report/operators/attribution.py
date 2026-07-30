"""Attribution 算子 — 1.增益归因。"""

import pandas as pd

from .._base import ReportOperator, SubSection


class AttributionOperator(ReportOperator):
    """增益归因 — 归因表（维度/绝对值/备注）。"""

    @property
    def name(self) -> str:
        return "attribution"

    @property
    def title(self) -> str:
        return "1.增益归因"

    def compute(self, context) -> list[SubSection]:
        rows = [
            {"维度": "样本", "绝对值": "", "备注": "结论需手动填写"},
            {"维度": "标签", "绝对值": "", "备注": "结论需手动填写"},
            {"维度": "特征", "绝对值": "", "备注": "结论需手动填写"},
            {"维度": "算法", "绝对值": "", "备注": "结论需手动填写"},
            {"维度": "其他", "绝对值": "", "备注": "结论需手动填写"},
        ]
        return [SubSection(self.title, pd.DataFrame(rows))]
