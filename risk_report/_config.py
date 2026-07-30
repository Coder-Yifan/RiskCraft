"""报告配置 — SheetConfig + DocumentConfig。

核心设计:
- SheetConfig 将算子结果分配到 sheet，实现算子与 Sheet 解耦
- DocumentConfig 组合多个 SheetConfig 形成完整文档配置
- 用户可自定义 DocumentConfig（选择部分算子、添加自定义算子）
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._base import ReportOperator
    from ._format import FormatConfig


@dataclass
class SheetConfig:
    """Sheet 配置: 将一组算子结果写入同一个 sheet。

    Parameters
    ----------
    sheet_name : str
        Excel sheet 名，如 "1.模型设计"
    operators : list[ReportOperator]
        该 sheet 包含的算子列表，按顺序排列
    """

    sheet_name: str
    operators: list["ReportOperator"]


@dataclass
class DocumentConfig:
    """文档配置: 组合多个 SheetConfig 形成完整报告。

    Parameters
    ----------
    sheets : list[SheetConfig]
        Sheet 配置列表，按顺序排列
    format_config : FormatConfig
        全局格式化配置
    """

    sheets: list[SheetConfig]
    format_config: "FormatConfig | None" = None
