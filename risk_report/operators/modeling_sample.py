"""ModelingSample 算子 — 1.5建模样本。"""

import pandas as pd

from .._base import ReportOperator, SubSection, placeholder_df


class ModelingSampleOperator(ReportOperator):
    """建模样本 — 样本分布表（训练集/测试集/验证集/压测集 好坏占比）。"""

    @property
    def name(self) -> str:
        return "modeling_sample"

    @property
    def title(self) -> str:
        return "1.5建模样本"

    def compute(self, context) -> list[SubSection]:
        stats = context.get_sample_stats()
        if not stats:
            return [SubSection(self.title, placeholder_df(
                "数据集未提供，请通过 ReportContext.data + tag_col + label_col 传入"
            ))]

        # 主标签样本分布
        rows = []
        total_goods = 0
        total_bads = 0
        total_n = 0
        for cn_name, s in stats.items():
            rows.append({
                "样本集": cn_name,
                "好样本": s["goods"],
                "坏样本": s["bads"],
                "灰样本": s["gray"],
                "总量": s["total"],
                "坏占比": s["bad_rate"],
            })
            total_goods += s["goods"]
            total_bads += s["bads"]
            total_n += s["total"]

        # 总计行
        rows.append({
            "样本集": "总计",
            "好样本": total_goods,
            "坏样本": total_bads,
            "灰样本": "",
            "总量": total_n,
            "坏占比": total_bads / total_n if total_n > 0 else 0,
        })

        subs = [SubSection("样本分布状况", pd.DataFrame(rows))]

        # extra_labels 压测集样本分布
        if context.extra_labels:
            extra_rows = []
            for label_col in context.extra_labels:
                extra_stats = context.get_sample_stats(label_col)
                for cn_name, s in extra_stats.items():
                    extra_rows.append({
                        "标签列": label_col,
                        "样本集": cn_name,
                        "好样本": s["goods"],
                        "坏样本": s["bads"],
                        "总量": s["total"],
                        "坏占比": s["bad_rate"],
                    })
            if extra_rows:
                subs.append(SubSection("压测集样本分布", pd.DataFrame(extra_rows)))

        return subs
