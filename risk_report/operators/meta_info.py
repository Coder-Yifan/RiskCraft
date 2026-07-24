"""MetaInfoOperator — 模型说明。"""

import pandas as pd

from .._base import ReportOperator, ReportSectionResult, SubSection
from .._context import ReportContext


class MetaInfoOperator(ReportOperator):
    """模型说明算子 — 产出元信息表。"""

    @property
    def name(self) -> str:
        return "meta_info"

    @property
    def title(self) -> str:
        return "模型说明"

    def compute(self, context: ReportContext) -> ReportSectionResult:
        rows = [
            {"项目": "模型名称", "内容": context.model_name or "未填写"},
            {"项目": "开发人员", "内容": context.developer or "未填写"},
            {"项目": "验证人员", "内容": context.validator or "未填写"},
            {"项目": "业务需求人员", "内容": context.business_owner or "未填写"},
            {"项目": "模型开发背景", "内容": context.background or "未填写"},
            {"项目": "模型应用场景", "内容": context.application or "未填写"},
            {"项目": "模型开发文档内容", "内容": "1.模型设计\n2.变量分析\n3.模型表现\n附件1-补充分析\n附件2-模型使用方案\n附件3-变量描述"},
        ]

        df = pd.DataFrame(rows)
        return ReportSectionResult(
            sheet_name=self.title,
            sub_sections=[SubSection(title="模型说明", data=df)],
        )
