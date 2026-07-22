"""
Lending Club 贷款数据集加载器 — LendingClubLoader

数据集来源：https://www.kaggle.com/datasets/wordsforthewise/lending-club

Lending Club 是美国最大的 P2P 借贷平台，该数据集包含 2007-2018 年
约 226 万条已接受贷款记录，涵盖 151 个特征，是风控建模领域最经典的
公开数据集之一。

目标变量：loan_status → 二值化
    - 正例（违约）: "Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"
    - 负例（正常）: "Fully Paid", "Does not meet the credit policy. Status:Fully Paid"
    - 其余类别（Current / Late / In Grace Period 等）默认剔除

典型用法:
    >>> loader = LendingClubLoader()
    >>> X, y = loader.load()           # 首次运行自动下载
    >>> X.shape, y.mean()
"""

import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


# ============================================================
# 常量
# ============================================================

# Kaggle 数据集标识
_KAGGLE_DATASET = "wordsforthewise/lending-club"

# 已接受贷款文件名（压缩包内）
_ACCEPTED_FILENAME = "accepted_2007_to_2018Q4.csv"

# 违约标签
_DEFAULT_LABELS = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
}

# 正常还款标签
_PAID_LABELS = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}

# 风控建模常用特征（从 151 列中精选，剔除数据泄露列和冗余列）
_SELECTED_FEATURES = {
    # ---- 借款人基本信息 ----
    "loan_amnt",           # 贷款金额
    "term",                # 贷款期限（36/60个月）
    "int_rate",            # 利率
    "installment",         # 月供
    "grade",               # 信用等级（A-G）
    "sub_grade",           # 细分等级
    "emp_length",          # 工作年限
    "home_ownership",      # 房产状态
    "annual_inc",          # 年收入
    "verification_status", # 收入验证状态
    # ---- 债务信息 ----
    "dti",                 # 债务收入比
    "revol_bal",           # 循环余额
    "revol_util",          # 循环额度使用率
    # ---- 信用历史 ----
    "delinq_2yrs",         # 近2年逾期次数
    "earliest_cr_line",    # 最早信用额度开户时间
    "fico_range_low",      # FICO 评分下限
    "fico_range_high",     # FICO 评分上限
    "open_acc",            # 在用信贷账户数
    "pub_rec",             # 公共记录数
    "total_acc",           # 信贷账户总数
    # ---- 贷款用途 ----
    "purpose",             # 贷款用途
    "addr_state",          # 所在州
    # ---- 逾期历史 ----
    "acc_now_delinq",      # 当前逾期账户数
    "tot_cur_bal",         # 当前总余额
    "tot_coll_amt",        # 总催收金额
    "total_rev_hi_lim",    # 循环额度上限
    "mort_acc",            # 抵押贷款账户数
    "num_actv_rev_tl",     # 活跃循环账户数
    "num_tl_90g_dpd_24m",  # 近24月90+逾期账户数
    "pct_tl_nvr_dlq",      # 从未逾期账户占比
    "pub_rec_bankruptcies", # 公共破产记录数
}

# 需要剔除的数据泄露特征（贷款发放后才产生的信息）
_LEAKAGE_FEATURES = {
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp",
    "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "last_pymnt_amnt",
    "last_pymnt_d", "last_fico_range_high", "last_fico_range_low",
    "last_credit_pull_d", "out_prncp", "out_prncp_inv",
    "next_pymnt_d", "debt_settlement_flag", "debt_settlement_flag_date",
    "settlement_status", "settlement_date", "settlement_amount",
    "settlement_percentage", "settlement_term",
    "hardship_flag", "hardship_type", "hardship_reason",
    "hardship_status", "hardship_start_date", "hardship_end_date",
    "hardship_amount", "hardship_length", "hardship_dpd",
    "hardship_loan_status", "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance", "hardship_last_payment_amount",
    "disbursement_method",
}


class LendingClubLoader(BaseEstimator):
    """
    Lending Club 贷款数据集加载器。

    自动下载、缓存、预处理 Lending Club 2007-2018 贷款数据，
    返回可直接用于风控建模的 (X, y)。

    Parameters
    ----------
    data_dir : str or Path, default="~/.risk_ml/datasets/lending_club"
        数据缓存目录。首次调用 load() 时下载数据到此目录，
        后续调用直接读取缓存。
    use_features : set[str] or "all" or "selected", default="selected"
        使用的特征列：
        - "selected": 仅使用风控建模精选特征（约 30 列）
        - "all": 使用全部特征（剔除数据泄露列）
        - 自定义 set: 传入特征名集合
    sample_ratio : float or None, default=None
        采样比例，用于快速实验。如 0.1 表示取 10% 数据。
        为 None 时不采样，加载全量数据。
    random_state : int or None, default=42
        采样随机种子。
    drop_leakage : bool, default=True
        是否自动剔除数据泄露特征（贷款发放后才产生的信息）。

    Example
    -------
    >>> loader = LendingClubLoader()
    >>> X, y = loader.load()
    >>> print(f"样本数: {len(X)}, 违约率: {y.mean():.4f}")

    >>> # 快速实验：仅取 5% 数据
    >>> loader = LendingClubLoader(sample_ratio=0.05)
    >>> X, y = loader.load()

    >>> # 使用全部特征
    >>> loader = LendingClubLoader(use_features="all")
    >>> X, y = loader.load()
    """

    def __init__(
        self,
        data_dir="~/.risk_ml/datasets/lending_club",
        use_features="selected",
        sample_ratio=None,
        random_state=42,
        drop_leakage=True,
    ):
        self.data_dir = data_dir
        self.use_features = use_features
        self.sample_ratio = sample_ratio
        self.random_state = random_state
        self.drop_leakage = drop_leakage

    def _resolve_data_dir(self):
        """解析数据目录路径。"""
        return Path(self.data_dir).expanduser().resolve()

    def _csv_path(self):
        """返回 CSV 文件路径。"""
        return self._resolve_data_dir() / _ACCEPTED_FILENAME

    def _zip_path(self):
        """返回下载的 ZIP 文件路径。"""
        return self._resolve_data_dir() / "lending-club.zip"

    def _download(self):
        """通过 Kaggle API 下载数据集。"""
        data_dir = self._resolve_data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)

        csv_path = self._csv_path()
        if csv_path.exists():
            return

        zip_path = self._zip_path()
        if not zip_path.exists():
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
            except ImportError:
                raise ImportError(
                    "需要 kaggle 包下载数据集。请执行:\n"
                    "  pip install kaggle\n"
                    "并将 API Key 放置到 ~/.kaggle/kaggle.json\n"
                    "获取地址: https://www.kaggle.com/settings → API → Create New Token"
                )

            print(f"[LendingClubLoader] 正在从 Kaggle 下载数据集 "
                  f"({_KAGGLE_DATASET})，约 1.1 GB ...")
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(
                _KAGGLE_DATASET,
                path=str(data_dir),
                unzip=False,
            )
            print("[LendingClubLoader] 下载完成。")

        # 解压（仅提取已接受贷款文件）
        if zip_path.exists() and not csv_path.exists():
            print(f"[LendingClubLoader] 正在解压 {_ACCEPTED_FILENAME} ...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                # 查找文件（Kaggle ZIP 内可能带目录前缀）
                candidates = [
                    n for n in zf.namelist()
                    if n.endswith(_ACCEPTED_FILENAME)
                ]
                if not candidates:
                    # 兜底：找最大的 CSV
                    csv_members = [n for n in zf.namelist() if n.endswith(".csv")]
                    candidates = [
                        max(csv_members, key=lambda n: zf.getinfo(n).file_size)
                    ]
                zf.extract(candidates[0], data_dir)
                # 移动到预期路径
                extracted = data_dir / candidates[0]
                if extracted != csv_path:
                    extracted.rename(csv_path)
            print("[LendingClubLoader] 解压完成。")

    def _make_target(self, df):
        """
        将 loan_status 二值化为目标变量。

        违约 = 1，正常 = 0，其余类别剔除。
        """
        mask_default = df["loan_status"].isin(_DEFAULT_LABELS)
        mask_paid = df["loan_status"].isin(_PAID_LABELS)
        mask_keep = mask_default | mask_paid

        y = pd.Series(np.nan, index=df.index, dtype=float)
        y[mask_default] = 1
        y[mask_paid] = 0

        return y, mask_keep

    def _select_features(self, df):
        """根据 use_features 配置选择特征列。"""
        if self.use_features == "selected":
            use_cols = _SELECTED_FEATURES & set(df.columns)
        elif self.use_features == "all":
            use_cols = set(df.columns)
        elif isinstance(self.use_features, set):
            use_cols = self.use_features & set(df.columns)
        else:
            raise ValueError(
                f"use_features 参数无效: {self.use_features}，"
                f"可选 'selected' / 'all' / 特征名集合"
            )

        # 剔除目标列和标识列
        use_cols -= {"loan_status", "id", "url", "member_id"}
        # 剔除数据泄露列
        if self.drop_leakage:
            use_cols -= _LEAKAGE_FEATURES

        return sorted(use_cols)

    def _preprocess(self, df):
        """基础预处理：类型转换。"""
        # term: " 36 months" → 36
        if "term" in df.columns and df["term"].dtype == object:
            df["term"] = df["term"].str.extract(r"(\d+)").astype(float)

        # emp_length: "< 1 year" → 0, "10+ years" → 10, etc.
        if "emp_length" in df.columns and df["emp_length"].dtype == object:
            _emp_map = {
                "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
                "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
                "8 years": 8, "9 years": 9, "10+ years": 10,
            }
            df["emp_length"] = df["emp_length"].map(_emp_map).astype(float)

        # revol_util: "15.6%" → 15.6
        if "revol_util" in df.columns and df["revol_util"].dtype == object:
            df["revol_util"] = (
                df["revol_util"].str.rstrip("%").astype(float)
            )

        # int_rate: "12.5%" → 12.5
        if "int_rate" in df.columns and df["int_rate"].dtype == object:
            df["int_rate"] = (
                df["int_rate"].str.rstrip("%").astype(float)
            )

        return df

    def load(self):
        """
        加载并预处理 Lending Club 数据集。

        首次调用时自动从 Kaggle 下载数据（约 1.1 GB 压缩包），
        后续调用直接读取本地缓存。

        Returns
        -------
        X : pandas.DataFrame
            特征矩阵。
        y : pandas.Series
            目标变量（0=正常还款，1=违约）。
        """
        # 确保数据已下载
        self._download()

        csv_path = self._csv_path()
        if not csv_path.exists():
            raise FileNotFoundError(
                f"数据文件不存在: {csv_path}\n"
                f"请确认下载是否成功。"
            )

        print(f"[LendingClubLoader] 正在加载 {csv_path} ...")
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"[LendingClubLoader] 原始数据: {df.shape[0]} 行, {df.shape[1]} 列")

        # 构造目标变量，剔除中间状态样本
        y, mask_keep = self._make_target(df)
        df = df[mask_keep].copy()
        y = y[mask_keep].copy()
        df.drop(columns=["loan_status"], inplace=True)
        print(f"[LendingClubLoader] 保留已结清样本: {len(df)} 行, "
              f"违约率: {y.mean():.4f}")

        # 选择特征
        use_cols = self._select_features(df)
        df = df[use_cols].copy()

        # 基础预处理
        df = self._preprocess(df)

        # 采样
        if self.sample_ratio is not None and self.sample_ratio < 1.0:
            n_sample = int(len(df) * self.sample_ratio)
            sample_idx = df.sample(
                n=n_sample, random_state=self.random_state
            ).index
            df = df.loc[sample_idx].copy()
            y = y.loc[sample_idx].copy()
            print(f"[LendingClubLoader] 采样 {self.sample_ratio:.0%}: "
                  f"{len(df)} 行")

        # 重置索引
        df.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)

        # 记录元信息
        self.n_samples_ = len(df)
        self.n_features_ = df.shape[1]
        self.feature_names_ = df.columns.tolist()
        self.default_rate_ = float(y.mean())

        return df, y

    def data_dictionary(self):
        """
        返回风控建模精选特征的数据字典。

        Returns
        -------
        dict
            特征名 → 中文描述的字典。
        """
        return {
            "loan_amnt": "贷款金额",
            "term": "贷款期限（月）",
            "int_rate": "利率（%）",
            "installment": "月供金额",
            "grade": "信用等级（A-G）",
            "sub_grade": "细分等级",
            "emp_length": "工作年限",
            "home_ownership": "房产状态",
            "annual_inc": "年收入",
            "verification_status": "收入验证状态",
            "dti": "债务收入比",
            "revol_bal": "循环余额",
            "revol_util": "循环额度使用率（%）",
            "delinq_2yrs": "近2年逾期次数",
            "earliest_cr_line": "最早信用额度开户时间",
            "fico_range_low": "FICO评分下限",
            "fico_range_high": "FICO评分上限",
            "open_acc": "在用信贷账户数",
            "pub_rec": "公共记录数",
            "total_acc": "信贷账户总数",
            "purpose": "贷款用途",
            "addr_state": "所在州",
            "acc_now_delinq": "当前逾期账户数",
            "tot_cur_bal": "当前总余额",
            "tot_coll_amt": "总催收金额",
            "total_rev_hi_lim": "循环额度上限",
            "mort_acc": "抵押贷款账户数",
            "num_actv_rev_tl": "活跃循环账户数",
            "num_tl_90g_dpd_24m": "近24月90+逾期账户数",
            "pct_tl_nvr_dlq": "从未逾期账户占比",
            "pub_rec_bankruptcies": "公共破产记录数",
        }
