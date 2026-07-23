"""Excel 格式化配置。"""

from dataclasses import dataclass


@dataclass
class FormatConfig:
    """Excel 输出格式化配置。

    Attributes
    ----------
    float_format : str
        浮点数 Excel 格式字符串
    integer_format : str
        整数 Excel 格式字符串
    percent_format : str
        百分比 Excel 格式字符串
    header_bg_color : str
        表头背景色（十六进制，不含 #）
    header_font_color : str
        表头字体色（十六进制，不含 #）
    title_font_size : int
        子章节标题字号
    title_font_bold : bool
        子章节标题是否加粗
    data_font_size : int
        数据行字号
    border_style : str
        边框样式（thin / medium / thick）
    subsection_gap_rows : int
        子章节之间的空行数
    """

    float_format: str = "#,##0.0000"
    integer_format: str = "#,##0"
    percent_format: str = "0.00%"
    header_bg_color: str = "4472C4"
    header_font_color: str = "FFFFFF"
    title_font_size: int = 12
    title_font_bold: bool = True
    data_font_size: int = 10
    border_style: str = "thin"
    subsection_gap_rows: int = 2


DEFAULT_FORMAT = FormatConfig()
