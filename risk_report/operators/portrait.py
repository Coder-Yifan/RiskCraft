"""Portrait 算子 — 4.模型画像表现。"""

import pandas as pd

from .._base import ReportOperator, SubSection, placeholder_df


class PortraitOperator(ReportOperator):
    """模型画像表现 — 画像分箱表（新模型+对标模型各一个 SubSection）。"""

    @property
    def name(self) -> str:
        return "portrait"

    @property
    def title(self) -> str:
        return "4.模型画像表现"

    def compute(self, context) -> list[SubSection]:
        if context.portrait_data is not None:
            return [SubSection(self.title, context.portrait_data, note="结论需手动填写")]

        return [SubSection(self.title, placeholder_df(
            "画像数据未提供，请通过 ReportContext.portrait_data 传入"
        ))]
