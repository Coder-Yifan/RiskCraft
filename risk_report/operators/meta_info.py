"""MetaInfo 算子 — 1.模型说明。

产出元信息表（模型名称/开发人员/验证人员/业务需求人员/开发背景/应用场景/文档内容）。
"""

import pandas as pd

from .._base import ReportOperator, SubSection


class MetaInfoOperator(ReportOperator):
    """模型说明 — 元信息汇总表。"""

    @property
    def name(self) -> str:
        return "meta_info"

    @property
    def title(self) -> str:
        return "1.模型说明"

    def compute(self, context) -> list[SubSection]:
        rows = [
            {"项目": "模型名称", "内容": context.model_name or "未填写"},
            {"项目": "开发人员", "内容": context.developer or "未填写"},
            {"项目": "验证人员", "内容": context.validator or "未填写"},
            {"项目": "业务需求人员", "内容": context.business_owner or "未填写"},
            {"项目": "开发背景", "内容": context.background or "未填写"},
            {"项目": "应用场景", "内容": context.application or "未填写"},
            {"项目": "文档内容", "内容": "1.模型设计\n2.变量分析\n3.模型表现\n附件1-补充分析\n附件2-模型使用方案\n附件3-变量描述"},
        ]
        df = pd.DataFrame(rows)
        return [SubSection(self.title, df)]
