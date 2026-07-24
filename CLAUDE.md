# RiskCraft - 项目指引

## 项目概述
RiskCraft 是一个风险建模与特征工程框架，包含两大子项目：
- **feature_derivative** — 多端兼容的特征衍生框架（Pandas/PySpark/Dict）
- **risk_ml** — sklearn 兼容的风控建模 ML 框架（清洗/分箱/WOE/筛选/Pipeline）
- **risk_report** — 模型报告自动产出模块（从 risk_ml.report 迁移为独立包）

## 代码风格
- Python 3.12，类型注解，中文注释
- feature_derivative: 策略模式，三端适配
- risk_ml: sklearn 基类继承（RiskTransformer / RiskSelector），fit/transform 接口

## Token 节约规则（必须遵守）
1. **修改已有文件时，必须使用 Edit 工具（局部替换），禁止使用 Write 工具重写整个文件**
   - 例外：仅当创建全新文件时才使用 Write
2. **Read 文件时，优先使用 offset+limit 参数只读需要的部分**，不要一次性读整个大文件
3. **不要在回复中重复粘贴大段代码**，引用 `file_path:line_number` 即可
4. **Git 提交时只 `git add` 变更的文件**，不要 `git add -A`
5. **并行执行独立的任务**，减少来回对话轮次

## 目录结构
```
RiskCraft/
├── feature_derivative/     # 特征衍生框架
│   ├── exceptions.py       # 自定义异常
│   ├── parser.py           # 表达式解析器
│   ├── sandbox.py          # 安全沙箱
│   ├── strategies.py       # 策略模式（Pandas/Spark/Online）
│   ├── context.py          # 上下文类
│   └── __init__.py         # 公共 API
├── risk_ml/                # 风控建模 ML 框架
│   ├── _base.py            # 基类：RiskTransformer / RiskSelector
│   ├── preprocessing/      # 特征清洗（FeatureCleaner）
│   ├── binning/            # 分箱（BaseBinner / ChiMergeBinner）
│   ├── encoding/           # WOE 编码（WoeEncoder / BinnerWoeEncoder）
│   ├── feature_selection/  # 特征筛选（IV/Correlation/PSI）
│   ├── estimator/          # 估计器（RiskXGBClassifier / OptunaTuner）
│   ├── dataset/            # 数据集加载器（LendingClubLoader）
│   └── tests/              # 测试套件
├── risk_report/            # 模型报告自动产出模块（独立包）
│   ├── __init__.py         # 公共 API
│   ├── _base.py            # 基类：ReportOperator / SubSection / ReportSectionResult
│   ├── _context.py         # 报告上下文：ReportContext / PipelineAttributes
│   ├── _excel.py           # Excel 写入器
│   ├── _format.py          # 格式配置
│   ├── _scoring.py         # 计算工具（lift/swap/ks/stats）
│   ├── report.py           # 组合器：ModelReport
│   ├── operators/          # 算子模块（11 算子）
│   ├── tests/              # 测试套件
│   └── demo_report.py      # 演示脚本
├── tests/                  # feature_derivative 测试
├── demo.py                 # 演示脚本
└── requirements.txt        # 依赖
```

## 运行环境
- Python: D:/softwares/conda/python.exe (3.12.4)
- 包管理: pip -i https://pypi.tuna.tsinghua.edu.cn/simple
