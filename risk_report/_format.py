"""Excel 格式化配置。"""

from dataclasses import dataclass, field


@dataclass
class FormatConfig:
    """Excel 输出格式化配置。

    Attributes
    ----------
    font_name : str
        全局字体名
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
    alt_row_color : str
        交替行背景色（十六进制，不含 #）
    alt_row_enabled : bool
        是否启用交替行色
    data_bar_columns : list[str]
        需要添加条件格式数据条的列名关键字
    percent_columns : list[str]
        需要显示为百分比的列名关键字
    """

    font_name: str = "微软雅黑"
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

    # 美化: 交替行色
    alt_row_color: str = "F2F7FB"
    alt_row_enabled: bool = True

    # 美化: 条件格式数据条
    data_bar_columns: list[str] = field(default_factory=lambda: [
        "lift", "cum_lift", "bad_rate", "gain_per", "weight_per",
    ])

    # 美化: 百分比列
    percent_columns: list[str] = field(default_factory=lambda: [
        "bad_rate", "total%", "坏占比", "缺失率",
    ])

    # 美化: 冻结表头的 Sheet（仅这些 Sheet 冻结首行，其他不冻结）
    freeze_header_sheets: list[str] = field(default_factory=lambda: [
        "2.变量分析", "附件3-变量描述",
    ])


DEFAULT_FORMAT = FormatConfig()
