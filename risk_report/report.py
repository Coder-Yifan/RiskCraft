"""ModelReport 组合器 — 配置驱动报告编排。

核心设计变更:
- ModelReport 由 DocumentConfig 配置驱动（8 Sheet / 22 算子）
- fit() 遍历 DocumentConfig.sheets，每个算子 compute() → list[SubSection]
- to_excel() 按 SheetConfig 将算子结果写入对应 sheet
- 支持自定义 DocumentConfig（选择部分算子、添加自定义算子）
"""

from typing import Self

from ._base import ReportOperator, SubSection
from ._config import DocumentConfig, SheetConfig
from ._context import ReportContext
from ._excel import ExcelWriter
from ._format import FormatConfig
from ._templates import DEFAULT_DOCUMENT_CONFIG


class ModelReport:
    """模型开发报告组合器。

    支持三种使用模式:
    1. 全量报告 — ModelReport().fit(context).to_excel(...)
    2. 模块化组装 — ModelReport(config=custom_config).fit(context).to_excel(...)
    3. 日常调用 — 单独调用算子的 compute() 方法

    Parameters
    ----------
    config : DocumentConfig | None
        报告配置。None 时使用 DEFAULT_DOCUMENT_CONFIG（标准模板，8 Sheet / 22 算子）。
    """

    def __init__(self, config: DocumentConfig | None = None):
        self.config = config or DEFAULT_DOCUMENT_CONFIG

    def fit(self, context: ReportContext) -> Self:
        """运行所有算子，存储结果。

        按 DocumentConfig.sheets 顺序遍历，每个算子 compute() → list[SubSection]。

        Parameters
        ----------
        context : ReportContext
            建模上下文

        Returns
        -------
        self
        """
        self.context = context
        # {算子name: list[SubSection]}
        self.results_: dict[str, list[SubSection]] = {}

        for sheet_cfg in self.config.sheets:
            for op in sheet_cfg.operators:
                self.results_[op.name] = op.compute(context)

        return self

    def to_excel(
        self,
        file_path: str,
        format_config: FormatConfig | None = None,
    ) -> None:
        """将所有结果写入 Excel。

        按 SheetConfig 将算子结果写入对应 sheet，
        同一 sheet 内的算子按配置顺序垂直排列。

        Parameters
        ----------
        file_path : str
            输出文件路径
        format_config : FormatConfig | None
            格式化配置（可选，覆盖默认）
        """
        fmt = format_config or self.config.format_config or None
        writer = ExcelWriter(file_path, format_config=fmt)

        # 按 SheetConfig 组装 sheet_results
        sheet_results: dict[str, list[SubSection]] = {}
        for sheet_cfg in self.config.sheets:
            sub_sections = []
            for op in sheet_cfg.operators:
                if op.name in self.results_:
                    sub_sections.extend(self.results_[op.name])
            if sub_sections:
                sheet_results[sheet_cfg.sheet_name] = sub_sections

        writer.write_report(sheet_results)
        writer.save()

    def get_result(self, name: str) -> list[SubSection]:
        """获取指定算子的结果。

        Parameters
        ----------
        name : str
            算子 name，如 'score_lift'

        Returns
        -------
        list[SubSection]
        """
        if name not in self.results_:
            raise KeyError(f"算子 '{name}' 不在结果中，可用: {list(self.results_.keys())}")
        return self.results_[name]
