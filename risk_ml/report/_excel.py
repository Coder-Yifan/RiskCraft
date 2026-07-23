"""独立 Excel 写入器 — 可灵活将任意 DataFrame 或报告章节落地为 Excel。"""

from typing import Any

import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from ._base import ReportSectionResult, SubSection
from ._format import FormatConfig, DEFAULT_FORMAT


class ExcelWriter:
    """独立 Excel 写入器。

    与 ModelReport 无耦合，支持三种写入方式:
    1. write_section() — 写入 ReportSectionResult（含多个 SubSection）
    2. write_dataframe() — 写入单个 DataFrame（可选标题）
    3. write_report() — 写入完整报告（多个 section → 多个 sheet）

    Parameters
    ----------
    file_path : str
        输出文件路径
    format_config : FormatConfig | None
        全局格式化配置，默认 DEFAULT_FORMAT
    auto_adjust_width : bool
        是否自动调整列宽，默认 True
    """

    def __init__(
        self,
        file_path: str,
        format_config: FormatConfig | None = None,
        auto_adjust_width: bool = True,
    ):
        self.file_path = file_path
        self.format_config = format_config or DEFAULT_FORMAT
        self.auto_adjust_width = auto_adjust_width
        self._wb = openpyxl.Workbook()
        # 删除默认 sheet
        if "Sheet" in self._wb.sheetnames:
            self._wb.remove(self._wb["Sheet"])

    def write_section(
        self,
        section: ReportSectionResult,
        sheet_name: str | None = None,
    ) -> None:
        """写入 ReportSectionResult 到一个 sheet。

        各 SubSection 垂直排列，标题加粗居中。

        Parameters
        ----------
        section : ReportSectionResult
            报告章节结果
        sheet_name : str | None
            sheet 名，默认取 section.sheet_name
        """
        name = sheet_name or section.sheet_name
        # 确保唯一 sheet 名
        if name in self._wb.sheetnames:
            name = f"{name}_{len(self._wb.sheetnames)}"
        ws = self._wb.create_sheet(name)

        fmt = section.format_config or self.format_config
        row = 1

        for sub in section.sub_sections:
            row = self._write_sub_section(ws, sub, row, fmt)
            row += fmt.subsection_gap_rows

    def write_dataframe(
        self,
        df: pd.DataFrame,
        sheet_name: str,
        start_row: int = 1,
        format_config: FormatConfig | None = None,
        title: str | None = None,
    ) -> None:
        """写入单个 DataFrame 到指定 sheet。

        Parameters
        ----------
        df : pd.DataFrame
            数据表
        sheet_name : str
            sheet 名
        start_row : int
            起始行号
        format_config : FormatConfig | None
            格式配置
        title : str | None
            标题（可选）
        """
        if sheet_name not in self._wb.sheetnames:
            ws = self._wb.create_sheet(sheet_name)
        else:
            ws = self._wb[sheet_name]

        fmt = format_config or self.format_config
        row = start_row

        if title:
            ws.cell(row=row, column=1, value=title).font = Font(
                size=fmt.title_font_size,
                bold=fmt.title_font_bold,
            )
            row += 1

        row = self._write_dataframe_rows(ws, df, row, fmt)

        if self.auto_adjust_width:
            self._adjust_column_width(ws)

    def write_report(self, sections: list[ReportSectionResult]) -> None:
        """写入完整报告（每个 section 一个 sheet）。"""
        for section in sections:
            self.write_section(section)

    def save(self) -> None:
        """保存到 file_path。"""
        self._wb.save(self.file_path)

    # ---- 内部方法 ----

    def _write_sub_section(
        self,
        ws: Worksheet,
        sub: SubSection,
        start_row: int,
        fmt: FormatConfig,
    ) -> int:
        """写入单个 SubSection，返回结束行号。"""
        row = start_row

        # 标题行
        title_cell = ws.cell(row=row, column=1, value=sub.title)
        title_cell.font = Font(size=fmt.title_font_size, bold=fmt.title_font_bold)
        row += 1

        # 备注（如有）
        if sub.note:
            ws.cell(row=row, column=1, value=sub.note).font = Font(size=9, italic=True)
            row += 1

        # 数据表
        row = self._write_dataframe_rows(ws, sub.data, row, fmt)
        row += 1

        if self.auto_adjust_width:
            self._adjust_column_width(ws)

        return row

    def _write_dataframe_rows(
        self,
        ws: Worksheet,
        df: pd.DataFrame,
        start_row: int,
        fmt: FormatConfig,
    ) -> int:
        """写入 DataFrame 表头+数据行，带格式化。"""
        row = start_row

        # 表头
        thin_border = Border(
            left=Side(style=fmt.border_style),
            right=Side(style=fmt.border_style),
            top=Side(style=fmt.border_style),
            bottom=Side(style=fmt.border_style),
        )
        header_fill = PatternFill(start_color=fmt.header_bg_color, end_color=fmt.header_bg_color, fill_type="solid")
        header_font = Font(color=fmt.header_font_color, bold=True, size=fmt.data_font_size)
        center_align = Alignment(horizontal="center", vertical="center", wrap_text=True)

        for col_idx, col_name in enumerate(df.columns, 1):
            cell = ws.cell(row=row, column=col_idx, value=col_name)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = center_align
            cell.border = thin_border
        row += 1

        # 数据行
        data_font = Font(size=fmt.data_font_size)
        data_align = Alignment(horizontal="center", vertical="center")

        for _, data_row in df.iterrows():
            for col_idx, val in enumerate(data_row, 1):
                # NaN → 空字符串
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    cell_value = ""
                else:
                    cell_value = val

                cell = ws.cell(row=row, column=col_idx, value=cell_value)
                cell.font = data_font
                cell.alignment = data_align
                cell.border = thin_border

                # 数字格式
                if isinstance(cell_value, (int, np.integer)):
                    cell.number_format = fmt.integer_format
                elif isinstance(cell_value, float) and not np.isnan(cell_value if isinstance(val, float) else 0):
                    # 判断百分比 vs 浮点
                    if isinstance(val, float) and abs(val) <= 1.0 and col_name.endswith("%") or col_name in ("bad_rate", "total%", "坏占比"):
                        cell.number_format = fmt.percent_format
                    else:
                        cell.number_format = fmt.float_format
            row += 1

        return row

    def _adjust_column_width(self, ws: Worksheet) -> None:
        """根据内容自动调整列宽。"""
        for col in ws.columns:
            max_length = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                try:
                    if cell.value:
                        # 中文字符宽度 ×2
                        val_str = str(cell.value)
                        length = len(val_str)
                        # 简单估算: 中文字符占2个宽度单位
                        cn_chars = sum(1 for c in val_str if '一' <= c <= '鿿')
                        length = length + cn_chars
                        max_length = max(max_length, length)
                except Exception:
                    pass
            adjusted_width = min(max(max_length + 2, 8), 50)
            ws.column_dimensions[col_letter].width = adjusted_width
