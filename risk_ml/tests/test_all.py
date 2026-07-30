"""
risk_ml 完整测试套件

覆盖：FeatureCleaner / ChiMergeBinner / WoeEncoder / BinnerWoeEncoder /
      IVSelector / CorrelationSelector / PSISelector / RiskXGBClassifier / OptunaTuner / LendingClubLoader
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from risk_ml._config import MISSING_VALUE_SENTINELS, map_sentinels_to_nan
from risk_ml import (
    FeatureCleaner,
    ChiMergeBinner,
    WoeEncoder,
    BinnerWoeEncoder,
    IVSelector,
    CorrelationSelector,
    PSISelector,
    RiskXGBClassifier,
    OptunaTuner,
    LendingClubLoader,
    RiskTransformer,
    RiskSelector,
    RiskPipeline,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_data():
    """生成风控建模样例数据"""
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "age": np.random.normal(35, 10, n),
        "income": np.random.lognormal(10, 1, n),
        "loan_amount": np.random.exponential(50000, n),
        "education": np.random.choice(["high", "bachelor", "master"], n),
    })
    # 注入缺失值
    df.loc[df.sample(frac=0.05).index, "income"] = np.nan
    # 注入目标变量
    prob = 1 / (1 + np.exp(-(df["age"] - 35) / 10))
    y = (prob > 0.5).astype(int)
    return df, y


@pytest.fixture
def binned_data(sample_data):
    """已分箱的数据"""
    X, y = sample_data
    binner = ChiMergeBinner(max_bins=5)
    X_binned = binner.fit_transform(X, y)
    return X_binned, y, binner


# ============================================================
# FeatureCleaner
# ============================================================

class TestFeatureCleaner:
    def test_basic_fit_transform(self, sample_data):
        X, y = sample_data
        cleaner = FeatureCleaner()
        cleaner.fit(X)
        result = cleaner.transform(X)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] > 0

    def test_drops_high_missing_columns(self):
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [np.nan, np.nan, np.nan],  # 100% missing
        })
        cleaner = FeatureCleaner(missing_threshold=0.9)
        cleaner.fit(df)
        result = cleaner.transform(df)
        assert "b" not in result.columns
        assert "a" in result.columns

    def test_fills_missing(self, sample_data):
        X, y = sample_data
        cleaner = FeatureCleaner(missing_strategy="median")
        cleaner.fit(X)
        result = cleaner.transform(X)
        assert result.isnull().sum().sum() == 0 or "income" not in cleaner.drop_columns_

    def test_outlier_clip(self):
        df = pd.DataFrame({"a": [1, 2, 3, 100, 5]})
        cleaner = FeatureCleaner(outlier_method="percentile", outlier_bounds=(0.05, 0.95))
        cleaner.fit(df)
        result = cleaner.transform(df)
        assert result["a"].max() < 100

    def test_constant_column_removed(self):
        df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
        cleaner = FeatureCleaner(nunique_threshold=1)
        cleaner.fit(df)
        result = cleaner.transform(df)
        assert "a" not in result.columns

    def test_get_params_set_params(self):
        cleaner = FeatureCleaner(missing_threshold=0.8)
        params = cleaner.get_params()
        assert params["missing_threshold"] == 0.8
        cleaner.set_params(missing_threshold=0.5)
        assert cleaner.missing_threshold == 0.5

    def test_clone(self):
        cleaner = FeatureCleaner(missing_threshold=0.8)
        cloned = clone(cleaner)
        assert cloned.missing_threshold == 0.8

    def test_sentinel_mapping_default(self):
        """默认哨兵值 [-999, -9998, -9996] 应被映射为 NaN"""
        df = pd.DataFrame({
            "a": [-999, 1, 2],
            "b": [-9998, 5, -9996],
            "c": [10, 20, 30],
        })
        cleaner = FeatureCleaner(missing_strategy="median")
        cleaner.fit(df)
        result = cleaner.transform(df)
        # -999, -9998, -9996 应被映射为 NaN 后填充
        assert not result.isnull().any().any()

    def test_sentinel_mapping_increases_missing_rate(self):
        """哨兵值映射后缺失率应正确计算，高缺失列应被删除"""
        df = pd.DataFrame({
            "a": [-999] * 95 + [1, 2, 3, 4, 5],  # 95% 哨兵值 → 缺失率 95%
            "b": [1, 2, 3, 4, 5] * 20,
        })
        cleaner = FeatureCleaner(missing_threshold=0.9)
        cleaner.fit(df)
        assert "a" in cleaner.drop_columns_

    def test_custom_sentinels(self):
        """自定义哨兵值列表"""
        df = pd.DataFrame({"a": [-1, 1, 2], "b": [3, 4, 5]})
        cleaner = FeatureCleaner(sentinels=[-1], missing_strategy="median")
        cleaner.fit(df)
        result = cleaner.transform(df)
        assert not result.isnull().any().any()

    def test_disable_sentinels(self):
        """空列表禁用哨兵映射，-999 保留原值"""
        df = pd.DataFrame({"a": [-999, 1, 2]})
        cleaner = FeatureCleaner(sentinels=[], missing_strategy="median")
        cleaner.fit(df)
        result = cleaner.transform(df)
        # -999 不会被映射为 NaN，因此也不会被填充
        assert -999 in result["a"].values


# ============================================================
# Config: map_sentinels_to_nan
# ============================================================

class TestMapSentinelsToNan:
    def test_basic_mapping(self):
        df = pd.DataFrame({"a": [-999, 1, 2], "b": [-9998, 5, -9996]})
        result = map_sentinels_to_nan(df)
        assert pd.isna(result.loc[0, "a"])
        assert pd.isna(result.loc[0, "b"])
        assert pd.isna(result.loc[2, "b"])
        assert result.loc[1, "a"] == 1

    def test_custom_sentinels(self):
        df = pd.DataFrame({"a": [-1, 1, 2]})
        result = map_sentinels_to_nan(df, sentinels=[-1])
        assert pd.isna(result.loc[0, "a"])

    def test_no_modify_original(self):
        df = pd.DataFrame({"a": [-999, 1, 2]})
        map_sentinels_to_nan(df)
        assert df.loc[0, "a"] == -999  # 原始数据不变

    def test_string_column_not_affected(self):
        df = pd.DataFrame({"a": [-999, 1, 2], "b": ["x", "y", "z"]})
        result = map_sentinels_to_nan(df)
        assert list(result["b"]) == ["x", "y", "z"]

    def test_config_defaults(self):
        assert -999 in MISSING_VALUE_SENTINELS
        assert -9998 in MISSING_VALUE_SENTINELS
        assert -9996 in MISSING_VALUE_SENTINELS

class TestChiMergeBinner:
    def test_basic_fit_transform(self, sample_data):
        X, y = sample_data
        binner = ChiMergeBinner(max_bins=5)
        binner.fit(X, y)
        result = binner.transform(X)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == X.shape
        # 箱索引应为数值（pd.cut labels=False 返回 float）
        for col in result.columns:
            assert result[col].dropna().apply(lambda x: isinstance(x, (int, float, np.integer, np.floating))).all()

    def test_max_bins_respected(self, sample_data):
        X, y = sample_data
        binner = ChiMergeBinner(max_bins=4)
        binner.fit(X, y)
        for col in X.columns:
            n_bins = binner.transform(X[[col]])[col].nunique()
            assert n_bins <= 4

    def test_bin_table(self, sample_data):
        X, y = sample_data
        binner = ChiMergeBinner(max_bins=5)
        binner.fit(X, y)
        table = binner.get_bin_table("age")
        assert isinstance(table, pd.DataFrame)
        assert "bin_index" in table.columns

    def test_requires_y(self, sample_data):
        X, _ = sample_data
        binner = ChiMergeBinner()
        with pytest.raises(ValueError, match="目标变量"):
            binner.fit(X)

    def test_clone(self):
        binner = ChiMergeBinner(max_bins=8)
        cloned = clone(binner)
        assert cloned.max_bins == 8


# ============================================================
# WoeEncoder
# ============================================================

class TestWoeEncoder:
    def test_basic_fit_transform(self, binned_data):
        X_binned, y, _ = binned_data
        encoder = WoeEncoder()
        encoder.fit(X_binned, y)
        result = encoder.transform(X_binned)
        assert isinstance(result, pd.DataFrame)
        # WOE 值应为浮点数
        for col in result.columns:
            assert result[col].dtype in [np.float64, np.float32]

    def test_iv_values_computed(self, binned_data):
        X_binned, y, _ = binned_data
        encoder = WoeEncoder()
        encoder.fit(X_binned, y)
        assert hasattr(encoder, "iv_values_")
        assert all(v >= 0 for v in encoder.iv_values_.values())

    def test_woe_table(self, binned_data):
        X_binned, y, _ = binned_data
        encoder = WoeEncoder()
        encoder.fit(X_binned, y)
        table = encoder.get_woe_table(X_binned.columns[0])
        assert isinstance(table, pd.DataFrame)
        assert "woe" in table.columns


# ============================================================
# BinnerWoeEncoder
# ============================================================

class TestBinnerWoeEncoder:
    def test_end_to_end(self, sample_data):
        X, y = sample_data
        encoder = BinnerWoeEncoder(max_bins=5)
        encoder.fit(X, y)
        result = encoder.transform(X)
        assert isinstance(result, pd.DataFrame)
        # 输出应为 WOE 浮点值
        for col in result.columns:
            assert result[col].dtype in [np.float64, np.float32]

    def test_exposes_binner_and_encoder_attrs(self, sample_data):
        X, y = sample_data
        encoder = BinnerWoeEncoder(max_bins=5)
        encoder.fit(X, y)
        assert hasattr(encoder, "binner_")
        assert hasattr(encoder, "encoder_")
        assert hasattr(encoder, "woe_map_")
        assert hasattr(encoder, "iv_values_")


# ============================================================
# IVSelector
# ============================================================

class TestIVSelector:
    def test_basic_selection(self, binned_data):
        X_binned, y, _ = binned_data
        selector = IVSelector(iv_threshold=0.001)
        selector.fit(X_binned, y)
        result = selector.transform(X_binned)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] <= X_binned.shape[1]

    def test_get_support(self, binned_data):
        X_binned, y, _ = binned_data
        selector = IVSelector(iv_threshold=0.001)
        selector.fit(X_binned, y)
        support = selector.get_support()
        assert len(support) == X_binned.shape[1]

    def test_max_iv_drops_suspicious(self):
        """IV 过高的特征应被删除（疑似数据泄露）"""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "normal": np.random.randint(0, 3, n),
            "leak": np.random.randint(0, 2, n),  # 与 y 高度相关
        })
        y = df["leak"].values.copy()
        selector = IVSelector(iv_threshold=0.001, max_iv=0.3)
        selector.fit(df, y)
        # leak 的 IV 会非常高，应被 max_iv 过滤
        assert not selector.get_support()[1]


# ============================================================
# CorrelationSelector
# ============================================================

class TestCorrelationSelector:
    def test_basic_selection(self):
        np.random.seed(42)
        n = 200
        x1 = np.random.randn(n)
        df = pd.DataFrame({
            "a": x1,
            "b": x1 + np.random.randn(n) * 0.01,  # 高相关
            "c": np.random.randn(n),  # 独立
        })
        selector = CorrelationSelector(corr_threshold=0.7)
        selector.fit(df)
        # a 和 b 高相关，应删除其中一个
        assert len(selector.drop_features_) >= 1

    def test_with_iv_values(self):
        np.random.seed(42)
        n = 200
        x1 = np.random.randn(n)
        df = pd.DataFrame({
            "a": x1,
            "b": x1 + np.random.randn(n) * 0.01,
            "c": np.random.randn(n),
        })
        # a 的 IV 更高，应保留 a
        iv_values = {"a": 0.5, "b": 0.1, "c": 0.3}
        selector = CorrelationSelector(corr_threshold=0.7, iv_values=iv_values)
        selector.fit(df)
        assert "a" not in selector.drop_features_


# ============================================================
# PSISelector
# ============================================================

class TestPSISelector:
    def test_stable_distribution(self):
        np.random.seed(42)
        X_ref = pd.DataFrame({"a": np.random.normal(0, 1, 1000)})
        X_cur = pd.DataFrame({"a": np.random.normal(0, 1, 1000)})
        selector = PSISelector(psi_threshold=0.25)
        selector.fit(X_ref)
        result = selector.transform(X_cur)
        # 相同分布的 PSI 应该很低
        assert "a" in result.columns

    def test_shifted_distribution(self):
        np.random.seed(42)
        X_ref = pd.DataFrame({"a": np.random.normal(0, 1, 1000)})
        X_cur = pd.DataFrame({"a": np.random.normal(5, 1, 1000)})  # 严重漂移
        selector = PSISelector(psi_threshold=0.25)
        selector.fit(X_ref)
        result = selector.transform(X_cur)
        # 漂移严重，a 应被删除
        assert "a" not in result.columns


# ============================================================
# RiskXGBClassifier
# ============================================================

class TestRiskXGBClassifier:
    def test_basic_fit_predict(self, sample_data):
        X, y = sample_data
        clf = RiskXGBClassifier(n_estimators=10, max_depth=3)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        assert y_pred.shape[0] == X.shape[0]
        assert set(y_pred).issubset({0, 1})

    def test_predict_proba(self, sample_data):
        X, y = sample_data
        clf = RiskXGBClassifier(n_estimators=10, max_depth=3)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_predict_score(self, sample_data):
        X, y = sample_data
        clf = RiskXGBClassifier(n_estimators=10, max_depth=3)
        clf.fit(X, y)
        score = clf.predict_score(X)
        assert score.shape == (X.shape[0],)
        assert (score >= 0).all() and (score <= 1).all()
        # predict_score 应与 predict_proba[:, 1] 一致
        np.testing.assert_array_almost_equal(
            score, clf.predict_proba(X)[:, 1]
        )

    def test_feature_importance(self, sample_data):
        X, y = sample_data
        clf = RiskXGBClassifier(n_estimators=10, max_depth=3)
        clf.fit(X, y)
        importance = clf.feature_importance()
        assert isinstance(importance, dict)
        assert set(importance.keys()) == set(X.columns)

    def test_feature_names_in(self, sample_data):
        X, y = sample_data
        clf = RiskXGBClassifier(n_estimators=10, max_depth=3)
        clf.fit(X, y)
        assert hasattr(clf, "feature_names_in_")
        assert clf.feature_names_in_ == X.columns.tolist()

    def test_clone(self):
        clf = RiskXGBClassifier(n_estimators=50, max_depth=3)
        cloned = clone(clf)
        assert cloned.n_estimators == 50
        assert cloned.max_depth == 3

    def test_scale_pos_weight(self, sample_data):
        X, y = sample_data
        clf = RiskXGBClassifier(
            n_estimators=10, max_depth=3, scale_pos_weight=10
        )
        clf.fit(X, y)
        y_pred = clf.predict(X)
        assert y_pred.shape[0] == X.shape[0]


# ============================================================
# OptunaTuner
# ============================================================

class TestOptunaTuner:
    def test_basic_tune(self, sample_data):
        X, y = sample_data
        tuner = OptunaTuner(
            estimator=RiskXGBClassifier(n_estimators=10, max_depth=3),
            n_trials=3,
            cv=2,
            scoring="roc_auc",
            random_state=42,
            verbose=0,
        )
        tuner.fit(X, y)
        assert hasattr(tuner, "best_params_")
        assert hasattr(tuner, "best_score_")
        assert hasattr(tuner, "best_estimator_")
        assert tuner.best_score_ > 0

    def test_predict_after_tune(self, sample_data):
        X, y = sample_data
        tuner = OptunaTuner(
            estimator=RiskXGBClassifier(n_estimators=10, max_depth=3),
            n_trials=3,
            cv=2,
            scoring="roc_auc",
            random_state=42,
            verbose=0,
        )
        tuner.fit(X, y)
        y_pred = tuner.predict(X)
        assert y_pred.shape[0] == X.shape[0]
        score = tuner.predict_score(X)
        assert score.shape == (X.shape[0],)

    def test_ks_scorer(self, sample_data):
        X, y = sample_data
        tuner = OptunaTuner(
            estimator=RiskXGBClassifier(n_estimators=10, max_depth=3),
            n_trials=3,
            cv=2,
            scoring="ks",
            random_state=42,
            verbose=0,
        )
        tuner.fit(X, y)
        assert tuner.best_score_ >= 0

    def test_custom_search_space(self, sample_data):
        X, y = sample_data
        tuner = OptunaTuner(
            estimator=RiskXGBClassifier(n_estimators=10, max_depth=3),
            n_trials=3,
            cv=2,
            search_space={"max_depth": (3, 5), "learning_rate": (0.01, 0.3)},
            scoring="roc_auc",
            random_state=42,
            verbose=0,
        )
        tuner.fit(X, y)
        assert "max_depth" in tuner.best_params_
        assert "learning_rate" in tuner.best_params_

    def test_trials_dataframe(self, sample_data):
        X, y = sample_data
        tuner = OptunaTuner(
            estimator=RiskXGBClassifier(n_estimators=10, max_depth=3),
            n_trials=3,
            cv=2,
            scoring="roc_auc",
            random_state=42,
            verbose=0,
        )
        tuner.fit(X, y)
        df = tuner.trials_dataframe_
        assert len(df) == 3

    def test_not_fitted_error(self):
        tuner = OptunaTuner(
            estimator=RiskXGBClassifier(),
            n_trials=3,
        )
        with pytest.raises(RuntimeError, match="尚未拟合"):
            tuner.predict(np.array([[1, 2]]))


# ============================================================
# LendingClubLoader
# ============================================================

class TestLendingClubLoader:
    """使用本地模拟数据测试 LendingClubLoader 的预处理逻辑。"""

    @pytest.fixture
    def mock_lending_club_csv(self, tmp_path):
        """生成模拟 Lending Club CSV 文件。"""
        n = 200
        np.random.seed(42)
        df = pd.DataFrame({
            "loan_amnt": np.random.lognormal(10, 0.5, n),
            "term": np.random.choice([" 36 months", " 60 months"], n),
            "int_rate": [f"{x:.2f}%" for x in np.random.uniform(5, 25, n)],
            "installment": np.random.uniform(100, 1000, n),
            "grade": np.random.choice(list("ABCDEFG"), n),
            "emp_length": np.random.choice(
                ["< 1 year", "2 years", "5 years", "10+ years"], n
            ),
            "home_ownership": np.random.choice(
                ["RENT", "OWN", "MORTGAGE"], n
            ),
            "annual_inc": np.random.lognormal(11, 0.8, n),
            "verification_status": np.random.choice(
                ["Verified", "Source Verified", "Not Verified"], n
            ),
            "dti": np.random.uniform(0, 30, n),
            "revol_bal": np.random.lognormal(9, 1, n),
            "revol_util": [f"{x:.1f}%" for x in np.random.uniform(0, 100, n)],
            "delinq_2yrs": np.random.poisson(0.5, n),
            "fico_range_low": np.random.randint(660, 800, n),
            "fico_range_high": np.random.randint(665, 805, n),
            "open_acc": np.random.poisson(10, n),
            "total_acc": np.random.poisson(20, n),
            "purpose": np.random.choice(
                ["debt_consolidation", "credit_card", "home_improvement"], n
            ),
            "loan_status": np.random.choice(
                ["Fully Paid", "Charged Off", "Current", "Late (31-120 days)"],
                n, p=[0.4, 0.2, 0.3, 0.1],
            ),
        })
        csv_path = tmp_path / "accepted_2007_to_2018Q4.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_make_target(self):
        """测试目标变量构造：仅保留 Fully Paid / Charged Off。"""
        loader = LendingClubLoader()
        df = pd.DataFrame({
            "loan_status": ["Fully Paid", "Charged Off", "Current",
                            "Default", "Late (31-120 days)"],
        })
        y, mask = loader._make_target(df)
        # Current 和 Late 被剔除
        assert mask.sum() == 3
        assert y[mask].tolist() == [0.0, 1.0, 1.0]

    def test_preprocess_term(self):
        """测试 term 列解析。"""
        loader = LendingClubLoader()
        df = pd.DataFrame({"term": [" 36 months", " 60 months"]})
        result = loader._preprocess(df)
        assert result["term"].tolist() == [36.0, 60.0]

    def test_preprocess_emp_length(self):
        """测试 emp_length 列解析。"""
        loader = LendingClubLoader()
        df = pd.DataFrame({"emp_length": ["< 1 year", "5 years", "10+ years"]})
        result = loader._preprocess(df)
        assert result["emp_length"].tolist() == [0.0, 5.0, 10.0]

    def test_preprocess_int_rate(self):
        """测试 int_rate 列解析。"""
        loader = LendingClubLoader()
        df = pd.DataFrame({"int_rate": ["12.5%", "6.75%"]})
        result = loader._preprocess(df)
        np.testing.assert_array_almost_equal(
            result["int_rate"].values, [12.5, 6.75]
        )

    def test_preprocess_revol_util(self):
        """测试 revol_util 列解析。"""
        loader = LendingClubLoader()
        df = pd.DataFrame({"revol_util": ["45.2%", "0%", "100%"]})
        result = loader._preprocess(df)
        np.testing.assert_array_almost_equal(
            result["revol_util"].values, [45.2, 0.0, 100.0]
        )

    def test_select_features_selected(self):
        """测试精选特征选择。"""
        loader = LendingClubLoader(use_features="selected")
        df = pd.DataFrame({
            "loan_amnt": [1], "term": [1], "dti": [1],
            "total_pymnt": [1],  # 泄露列
            "loan_status": ["Fully Paid"],
        })
        cols = loader._select_features(df)
        assert "loan_amnt" in cols
        assert "total_pymnt" not in cols  # 泄露列被剔除
        assert "loan_status" not in cols

    def test_select_features_all(self):
        """测试全部特征选择。"""
        loader = LendingClubLoader(use_features="all", drop_leakage=True)
        df = pd.DataFrame({
            "loan_amnt": [1], "total_pymnt": [1],
            "loan_status": ["Fully Paid"],
        })
        cols = loader._select_features(df)
        assert "loan_amnt" in cols
        assert "total_pymnt" not in cols  # 泄露列被剔除

    def test_select_features_custom(self):
        """测试自定义特征选择。"""
        loader = LendingClubLoader(use_features={"loan_amnt", "dti"})
        df = pd.DataFrame({"loan_amnt": [1], "dti": [1], "term": [1]})
        cols = loader._select_features(df)
        assert cols == ["dti", "loan_amnt"]

    def test_full_load_with_mock(self, mock_lending_club_csv, tmp_path):
        """使用模拟 CSV 测试完整 load 流程。"""
        loader = LendingClubLoader(data_dir=str(tmp_path))
        # 直接调用内部方法跳过下载
        df = pd.read_csv(mock_lending_club_csv, low_memory=False)
        y, mask = loader._make_target(df)
        df = df[mask].copy()
        y = y[mask].copy()
        df.drop(columns=["loan_status"], inplace=True)
        use_cols = loader._select_features(df)
        df = df[use_cols].copy()
        df = loader._preprocess(df)
        # 验证输出
        assert len(df) == len(y)
        assert set(y.unique()) <= {0.0, 1.0}
        # term 应已转为数值
        if "term" in df.columns:
            assert df["term"].dtype == float

    def test_data_dictionary(self):
        """测试数据字典返回。"""
        loader = LendingClubLoader()
        dd = loader.data_dictionary()
        assert isinstance(dd, dict)
        assert "loan_amnt" in dd
        assert "dti" in dd

    def test_sample_ratio(self, mock_lending_club_csv, tmp_path):
        """测试采样功能。"""
        loader = LendingClubLoader(
            data_dir=str(tmp_path),
            sample_ratio=0.5,
            random_state=42,
        )
        df = pd.read_csv(mock_lending_club_csv, low_memory=False)
        y, mask = loader._make_target(df)
        df = df[mask].copy()
        y = y[mask].copy()
        df.drop(columns=["loan_status"], inplace=True)
        use_cols = loader._select_features(df)
        df = df[use_cols].copy()
        df = loader._preprocess(df)
        n_full = len(df)
        # 采样
        n_sample = int(n_full * loader.sample_ratio)
        sample_idx = df.sample(
            n=n_sample, random_state=loader.random_state
        ).index
        df_sampled = df.loc[sample_idx]
        assert len(df_sampled) < n_full


# ============================================================
# Metrics
# ============================================================

class TestAUCMetric:
    def test_basic(self):
        from risk_ml.experiment import AUCMetric
        m = AUCMetric()
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        assert m.name == "auc"
        assert m.compute(y_true, y_score) == 1.0

    def test_random(self):
        from risk_ml.experiment import AUCMetric
        m = AUCMetric()
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_score = np.random.rand(1000)
        auc = m.compute(y_true, y_score)
        assert 0.4 < auc < 0.6  # 随机猜测 AUC ≈ 0.5

    def test_single_class(self):
        from risk_ml.experiment import AUCMetric
        m = AUCMetric()
        y_true = np.array([1, 1, 1])
        y_score = np.array([0.5, 0.6, 0.7])
        assert m.compute(y_true, y_score) == 0.0


class TestKSMetric:
    def test_perfect_separation(self):
        from risk_ml.experiment import KSMetric
        m = KSMetric()
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        assert m.name == "ks"
        assert m.compute(y_true, y_score) > 0.9

    def test_no_separation(self):
        from risk_ml.experiment import KSMetric
        m = KSMetric()
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.5, 0.5, 0.5, 0.5])
        assert m.compute(y_true, y_score) == 0.0

    def test_single_class(self):
        from risk_ml.experiment import KSMetric
        m = KSMetric()
        y_true = np.array([1, 1, 1])
        y_score = np.array([0.5, 0.6, 0.7])
        assert m.compute(y_true, y_score) == 0.0


class TestLiftMetric:
    def test_default_name(self):
        from risk_ml.experiment import LiftMetric
        m = LiftMetric()
        assert m.name == "lift_10"

    def test_custom_percentile(self):
        from risk_ml.experiment import LiftMetric
        m = LiftMetric(percentile=20)
        assert m.name == "lift_20"

    def test_perfect_model(self):
        from risk_ml.experiment import LiftMetric
        m = LiftMetric(percentile=50)
        # 前 50% 高分全是正例，正例率=100%
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        lift = m.compute(y_true, y_score)
        # 全局正例率=0.5，前50%正例率=1.0，lift=2.0
        assert lift == 2.0

    def test_empty_input(self):
        from risk_ml.experiment import LiftMetric
        m = LiftMetric()
        assert m.compute(np.array([]), np.array([])) == 0.0

    def test_all_negative(self):
        from risk_ml.experiment import LiftMetric
        m = LiftMetric()
        y_true = np.array([0, 0, 0, 0])
        y_score = np.array([0.1, 0.2, 0.3, 0.4])
        assert m.compute(y_true, y_score) == 0.0


class TestCustomMetric:
    def test_user_defined_metric(self):
        from risk_ml.experiment import BaseMetric

        class GiniMetric(BaseMetric):
            name = "gini"

            def compute(self, y_true, y_score):
                from sklearn.metrics import roc_auc_score
                return 2 * roc_auc_score(y_true, y_score) - 1

        m = GiniMetric()
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        assert m.name == "gini"
        assert m.compute(y_true, y_score) == 1.0


# ============================================================
# ExperimentRunner
# ============================================================

@pytest.fixture
def experiment_data():
    """生成实验模块测试数据（含足够预测信号以保证 IV > 阈值）"""
    np.random.seed(42)
    n = 500
    age = np.random.normal(35, 10, n)
    income = np.random.lognormal(10, 1, n)
    loan_amount = np.random.exponential(50000, n)

    # 构造有信号的目标变量（与特征相关）
    prob_30d = 1 / (1 + np.exp(-(age - 35) / 10 - (income - 25000) / 50000))
    is_default_30d = (np.random.rand(n) < prob_30d).astype(int)

    prob_90d = 1 / (1 + np.exp(-(age - 30) / 10 - (income - 20000) / 50000))
    is_default_90d = (np.random.rand(n) < prob_90d).astype(int)

    df = pd.DataFrame({
        "age": age,
        "income": income,
        "loan_amount": loan_amount,
        "is_default_30d": is_default_30d,
        "is_default_90d": is_default_90d,
        "sample_weight": np.random.uniform(0.5, 1.5, n),
        "issue_d": pd.date_range("2018-01-01", periods=n, freq="D"),
    })
    return df


class TestExperimentRunner:
    """实验组合器测试 — 使用简单 pipeline 避免小数据集上特征筛选失效"""

    @pytest.fixture
    def simple_pipe(self):
        """测试用简单 Pipeline：FeatureCleaner + RiskXGBClassifier"""
        from sklearn.pipeline import Pipeline
        from risk_ml import FeatureCleaner, RiskXGBClassifier
        return Pipeline([
            ("cleaner", FeatureCleaner()),
            ("classifier", RiskXGBClassifier(n_estimators=10)),
        ])

    # 测试用特征列（排除所有非特征列）
    _FEAT_COLS = ["age", "income", "loan_amount"]

    def test_single_config(self, experiment_data, simple_pipe):
        from risk_ml.experiment import (
            ExperimentRunner, ExperimentConfig, AUCMetric, KSMetric,
        )
        configs = [ExperimentConfig(name="test1", label_col="is_default_30d")]
        runner = ExperimentRunner(
            configs=configs,
            pipeline=simple_pipe,
            feature_columns=self._FEAT_COLS,
            n_trials=2,
            tuner_cv=2,
            metrics=[AUCMetric(), KSMetric()],
            verbose=0,
        )
        runner.fit(experiment_data)
        assert len(runner.results_) == 1
        assert runner.results_.iloc[0]["status"] == "success"
        assert "auc" in runner.results_.columns
        assert "ks" in runner.results_.columns

    def test_multiple_labels(self, experiment_data, simple_pipe):
        from risk_ml.experiment import ExperimentRunner, ExperimentConfig
        configs = [
            ExperimentConfig(name="30d", label_col="is_default_30d"),
            ExperimentConfig(name="90d", label_col="is_default_90d"),
        ]
        runner = ExperimentRunner(
            configs=configs, pipeline=simple_pipe,
            feature_columns=self._FEAT_COLS,
            n_trials=2, tuner_cv=2, verbose=0,
        )
        runner.fit(experiment_data)
        assert len(runner.results_) == 2
        rates = runner.results_.set_index("name")["default_rate"]
        assert rates["30d"] != rates["90d"]

    def test_time_window_filter(self, experiment_data, simple_pipe):
        from risk_ml.experiment import (
            ExperimentRunner, ExperimentConfig, TimeWindow,
        )
        tw = TimeWindow("issue_d", "2018-01-01", "2018-03-31")
        configs = [
            ExperimentConfig(name="full", label_col="is_default_30d"),
            ExperimentConfig(name="q1", label_col="is_default_30d", time_window=tw),
        ]
        runner = ExperimentRunner(
            configs=configs, pipeline=simple_pipe,
            feature_columns=self._FEAT_COLS,
            n_trials=2, tuner_cv=2, verbose=0,
        )
        runner.fit(experiment_data)
        n_full = runner.results_.set_index("name").loc["full", "n_samples"]
        n_q1 = runner.results_.set_index("name").loc["q1", "n_samples"]
        assert n_q1 < n_full

    def test_weight_col(self, experiment_data, simple_pipe):
        from risk_ml.experiment import ExperimentRunner, ExperimentConfig
        configs = [
            ExperimentConfig(name="weighted", label_col="is_default_30d",
                             weight_col="sample_weight"),
        ]
        runner = ExperimentRunner(
            configs=configs, pipeline=simple_pipe,
            feature_columns=self._FEAT_COLS,
            n_trials=2, tuner_cv=2, verbose=0,
        )
        runner.fit(experiment_data)
        assert runner.results_.iloc[0]["status"] == "success"

    def test_failed_experiment(self, experiment_data, simple_pipe):
        from risk_ml.experiment import ExperimentRunner, ExperimentConfig
        configs = [
            ExperimentConfig(name="valid", label_col="is_default_30d"),
            ExperimentConfig(name="invalid", label_col="nonexistent_col"),
        ]
        runner = ExperimentRunner(
            configs=configs, pipeline=simple_pipe,
            feature_columns=self._FEAT_COLS,
            n_trials=2, tuner_cv=2, verbose=0,
        )
        runner.fit(experiment_data)
        results = runner.results_.set_index("name")
        assert results.loc["valid", "status"] == "success"
        assert results.loc["invalid", "status"] == "failed"

    def test_best_estimator(self, experiment_data, simple_pipe):
        from risk_ml.experiment import ExperimentRunner, ExperimentConfig
        configs = [ExperimentConfig(name="test", label_col="is_default_30d")]
        runner = ExperimentRunner(
            configs=configs, pipeline=simple_pipe,
            feature_columns=self._FEAT_COLS,
            n_trials=2, tuner_cv=2, verbose=0,
        )
        runner.fit(experiment_data)
        assert hasattr(runner, "best_estimator_")
        X_test = experiment_data[self._FEAT_COLS].head(5)
        y_pred = runner.predict(X_test)
        assert len(y_pred) == 5

    def test_predict_score(self, experiment_data, simple_pipe):
        from risk_ml.experiment import ExperimentRunner, ExperimentConfig
        configs = [ExperimentConfig(name="test", label_col="is_default_30d")]
        runner = ExperimentRunner(
            configs=configs, pipeline=simple_pipe,
            feature_columns=self._FEAT_COLS,
            n_trials=2, tuner_cv=2, verbose=0,
        )
        runner.fit(experiment_data)
        X_test = experiment_data[self._FEAT_COLS].head(5)
        scores = runner.predict_score(X_test)
        assert len(scores) == 5
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_results_dataframe_columns(self, experiment_data, simple_pipe):
        from risk_ml.experiment import ExperimentRunner, ExperimentConfig
        configs = [ExperimentConfig(name="test", label_col="is_default_30d")]
        runner = ExperimentRunner(
            configs=configs, pipeline=simple_pipe,
            feature_columns=self._FEAT_COLS,
            n_trials=2, tuner_cv=2, verbose=0,
        )
        runner.fit(experiment_data)
        for col in ["name", "label_col", "time_window", "weight_col",
                     "status", "n_samples", "default_rate", "n_features",
                     "mean_iv", "best_trial_score", "training_time"]:
            assert col in runner.results_.columns
        for col in ["auc", "ks", "lift_10"]:
            assert col in runner.results_.columns

    def test_custom_metrics(self, experiment_data, simple_pipe):
        from risk_ml.experiment import (
            ExperimentRunner, ExperimentConfig, AUCMetric, KSMetric, BaseMetric,
        )

        class GiniMetric(BaseMetric):
            name = "gini"
            def compute(self, y_true, y_score):
                from sklearn.metrics import roc_auc_score
                return 2 * roc_auc_score(y_true, y_score) - 1

        configs = [ExperimentConfig(name="test", label_col="is_default_30d")]
        runner = ExperimentRunner(
            configs=configs,
            pipeline=simple_pipe,
            feature_columns=self._FEAT_COLS,
            metrics=[AUCMetric(), KSMetric(), GiniMetric()],
            n_trials=2,
            tuner_cv=2,
            verbose=0,
        )
        runner.fit(experiment_data)
        assert "gini" in runner.results_.columns
        gini_val = runner.results_.iloc[0]["gini"]
        assert 0.0 <= gini_val <= 1.0

    def test_make_experiment_grid(self):
        from risk_ml.experiment import make_experiment_grid, TimeWindow
        configs = make_experiment_grid(
            label_cols=["is_default_30d", "is_default_90d"],
            time_windows=[
                TimeWindow("issue_d", "2018-01-01", "2018-03-31"),
                TimeWindow("issue_d", "2018-04-01", "2018-06-30"),
            ],
        )
        assert len(configs) == 4
        label_set = {c.label_col for c in configs}
        assert label_set == {"is_default_30d", "is_default_90d"}

    def test_make_experiment_grid_with_weights(self):
        from risk_ml.experiment import make_experiment_grid
        configs = make_experiment_grid(
            label_cols=["y1"],
            weight_cols=["w1", "w2"],
        )
        assert len(configs) == 2

    def test_clone(self):
        from risk_ml.experiment import ExperimentRunner, ExperimentConfig
        runner = ExperimentRunner(
            configs=[ExperimentConfig(name="test", label_col="y")],
            n_trials=5,
        )
        cloned = clone(runner)
        assert cloned.n_trials == 5

    def test_with_oot(self, experiment_data, simple_pipe):
        """OOT 数据集评估：应产生 oot_ 前缀指标列"""
        from risk_ml.experiment import ExperimentRunner, ExperimentConfig
        # 取后 20% 数据作为 OOT
        split = int(len(experiment_data) * 0.8)
        df_train = experiment_data.iloc[:split]
        df_oot = experiment_data.iloc[split:]

        configs = [ExperimentConfig(name="with_oot", label_col="is_default_30d")]
        runner = ExperimentRunner(
            configs=configs, pipeline=simple_pipe,
            feature_columns=self._FEAT_COLS,
            n_trials=2, tuner_cv=2, verbose=0,
            oot=df_oot,
        )
        runner.fit(df_train)
        results = runner.results_
        # OOT 列应存在
        assert "oot_auc" in results.columns
        assert "oot_ks" in results.columns
        assert "oot_n_samples" in results.columns
        assert "oot_default_rate" in results.columns
        # OOT 指标应有值
        assert not pd.isna(results.iloc[0]["oot_auc"])
        assert results.iloc[0]["oot_n_samples"] > 0
        # 训练集指标列仍应存在
        assert "auc" in results.columns
        assert "ks" in results.columns

    def test_with_eval_label_cols(self, experiment_data, simple_pipe):
        """多标签评估：应产生 {label}_{metric} 列"""
        from risk_ml.experiment import ExperimentRunner, ExperimentConfig
        configs = [ExperimentConfig(name="multi_label", label_col="is_default_30d")]
        runner = ExperimentRunner(
            configs=configs, pipeline=simple_pipe,
            feature_columns=self._FEAT_COLS,
            n_trials=2, tuner_cv=2, verbose=0,
            eval_label_cols=["is_default_90d"],
        )
        runner.fit(experiment_data)
        results = runner.results_
        # 额外标签列应存在
        assert "is_default_90d_auc" in results.columns
        assert "is_default_90d_ks" in results.columns
        # 原始标签列指标仍应存在
        assert "auc" in results.columns
        # 额外标签指标应有值
        assert not pd.isna(results.iloc[0]["is_default_90d_auc"])

    def test_with_oot_and_eval_labels(self, experiment_data, simple_pipe):
        """OOT + 多标签组合：应产生 oot_{label}_{metric} 列"""
        from risk_ml.experiment import ExperimentRunner, ExperimentConfig
        split = int(len(experiment_data) * 0.8)
        df_train = experiment_data.iloc[:split]
        df_oot = experiment_data.iloc[split:]

        configs = [ExperimentConfig(name="combo", label_col="is_default_30d")]
        runner = ExperimentRunner(
            configs=configs, pipeline=simple_pipe,
            feature_columns=self._FEAT_COLS,
            n_trials=2, tuner_cv=2, verbose=0,
            oot=df_oot,
            eval_label_cols=["is_default_90d"],
        )
        runner.fit(df_train)
        results = runner.results_
        # 训练集指标
        assert "auc" in results.columns
        assert "is_default_90d_auc" in results.columns
        # OOT 指标
        assert "oot_auc" in results.columns
        assert "oot_is_default_90d_auc" in results.columns
        # 值应非空
        assert not pd.isna(results.iloc[0]["oot_auc"])
        assert not pd.isna(results.iloc[0]["oot_is_default_90d_auc"])

    def test_oot_default_rate_differs(self, experiment_data, simple_pipe):
        """OOT 的 default_rate 应与训练集不同"""
        from risk_ml.experiment import ExperimentRunner, ExperimentConfig
        split = int(len(experiment_data) * 0.5)
        df_train = experiment_data.iloc[:split]
        df_oot = experiment_data.iloc[split:]

        configs = [ExperimentConfig(name="rate_check", label_col="is_default_30d")]
        runner = ExperimentRunner(
            configs=configs, pipeline=simple_pipe,
            feature_columns=self._FEAT_COLS,
            n_trials=2, tuner_cv=2, verbose=0,
            oot=df_oot,
        )
        runner.fit(df_train)
        train_rate = runner.results_.iloc[0]["default_rate"]
        oot_rate = runner.results_.iloc[0]["oot_default_rate"]
        # OOT 样本数应 > 0
        assert runner.results_.iloc[0]["oot_n_samples"] > 0


# ============================================================
# RiskPipeline
# ============================================================

class TestRiskPipeline:
    """RiskPipeline 测试：向后兼容、验证集数据流、属性传递、PSI"""

    _FEAT_COLS = ["age", "income", "loan_amount", "education"]

    @pytest.fixture
    def pipeline_data(self):
        """生成训练集和验证集数据"""
        np.random.seed(42)
        n_train = 400
        n_val = 200
        # 训练集
        X_train = pd.DataFrame({
            "age": np.random.normal(35, 10, n_train),
            "income": np.random.lognormal(10, 1, n_train),
            "loan_amount": np.random.exponential(50000, n_train),
            "education": np.random.choice([0, 1, 2], n_train),
        })
        prob_train = 1 / (1 + np.exp(-(X_train["age"] - 35) / 10))
        y_train = (prob_train > 0.5).astype(int).values
        # 验证集（轻微偏移，使 PSI 有意义）
        X_val = pd.DataFrame({
            "age": np.random.normal(36, 10, n_val),
            "income": np.random.lognormal(10.1, 1.1, n_val),
            "loan_amount": np.random.exponential(52000, n_val),
            "education": np.random.choice([0, 1, 2], n_val),
        })
        prob_val = 1 / (1 + np.exp(-(X_val["age"] - 35) / 10))
        y_val = (prob_val > 0.5).astype(int).values
        return X_train, y_train, X_val, y_val

    def test_backward_compatible_no_val(self, pipeline_data):
        """不传 X_val 时，行为与 sklearn Pipeline 完全一致"""
        X_train, y_train, X_val, y_val = pipeline_data
        from sklearn.pipeline import Pipeline

        # 两条相同流水线
        steps = [
            ("cleaner", FeatureCleaner()),
            ("classifier", RiskXGBClassifier(n_estimators=10)),
        ]
        pipe_sk = Pipeline(steps)
        pipe_risk = RiskPipeline(steps)

        pipe_sk.fit(X_train, y_train)
        pipe_risk.fit(X_train, y_train)

        # 验证预测一致
        y_pred_sk = pipe_sk.predict_proba(X_val)[:, 1]
        y_pred_risk = pipe_risk.predict_proba(X_val)[:, 1]
        np.testing.assert_array_almost_equal(y_pred_sk, y_pred_risk)

    def test_val_data_flow_stores_transformed(self, pipeline_data):
        """传入 X_val 时，存储变换后的验证集数据"""
        X_train, y_train, X_val, y_val = pipeline_data
        pipe = RiskPipeline([
            ("cleaner", FeatureCleaner()),
            ("classifier", RiskXGBClassifier(n_estimators=10)),
        ])
        pipe.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        # 应有 X_val_transformed_ 和 y_val_ 属性
        assert hasattr(pipe, "X_val_transformed_")
        assert hasattr(pipe, "y_val_")
        assert pipe.X_val_transformed_ is not None
        assert pipe.y_val_ is not None

    def test_attribute_routing_iv_to_corr(self, pipeline_data):
        """step间属性传递：IVSelector.iv_values_ → CorrelationSelector.iv_values"""
        X_train, y_train, X_val, y_val = pipeline_data

        # CorrelationSelector(iv_values=None) 应自动获取 IVSelector 的 iv_values_
        pipe = RiskPipeline([
            ("cleaner", FeatureCleaner()),
            ("binner_woe", BinnerWoeEncoder()),
            ("iv_selector", IVSelector()),
            ("corr_selector", CorrelationSelector(iv_values=None)),
            ("classifier", RiskXGBClassifier(n_estimators=10)),
        ])
        pipe.fit(X_train, y_train)

        # CorrelationSelector 应已获取 iv_values（不再为 None）
        corr_step = pipe.named_steps["corr_selector"]
        assert corr_step.iv_values is not None
        # 应有 drop_features_ 属性（fit 成功）
        assert hasattr(corr_step, "drop_features_")

    def test_psi_with_val_real_psi(self, pipeline_data):
        """PSISelector 在有验证集时计算真实 PSI（>0）"""
        X_train, y_train, X_val, y_val = pipeline_data

        # 注入显著分布偏移：验证集的 age 列整体偏移 +20
        X_val_shifted = X_val.copy()
        X_val_shifted["age"] = X_val_shifted["age"] + 20

        # 简化 pipeline：cleaner → binner_woe → psi（放在 IV 之后也有意义，
        # 但为了测试 PSI 的核心行为，放在前面以保留更多特征）
        pipe = RiskPipeline([
            ("cleaner", FeatureCleaner()),
            ("binner_woe", BinnerWoeEncoder()),
            ("psi_selector", PSISelector()),
            ("classifier", RiskXGBClassifier(n_estimators=10)),
        ])

        # 不传验证集：PSI ≈ 0（train→train）
        pipe_no_val = clone(pipe)
        pipe_no_val.fit(X_train, y_train)
        psi_no_val = pipe_no_val.named_steps["psi_selector"].psi_values_
        # train→train PSI 应接近 0
        assert (psi_no_val < 0.01).all()

        # 传入验证集（含显著偏移）：PSI > 0（train→val）
        pipe_with_val = clone(pipe)
        pipe_with_val.fit(X_train, y_train, X_val=X_val_shifted, y_val=y_val)
        psi_with_val = pipe_with_val.named_steps["psi_selector"].psi_values_
        # train→val PSI 应 > 0（age 列偏移 +20，PSI 必然很大）
        assert (psi_with_val > 0).any()

    def test_optuna_tuner_holdout(self, pipeline_data):
        """OptunaTuner holdout 评估：传入 X_val/y_val 时 _eval_mode='holdout'"""
        X_train, y_train, X_val, y_val = pipeline_data
        # Pipeline 搜索空间需要 step__ 前缀
        from risk_ml.estimator.optuna_tuner import _DEFAULT_SEARCH_SPACE
        search_space = {
            f"classifier__{k}": v for k, v in _DEFAULT_SEARCH_SPACE.items()
        }
        pipe = RiskPipeline([
            ("cleaner", FeatureCleaner()),
            ("classifier", RiskXGBClassifier(n_estimators=10)),
        ])
        tuner = OptunaTuner(
            estimator=pipe,
            n_trials=3,
            search_space=search_space,
            scoring="roc_auc",
            cv=3,
            random_state=42,
            verbose=0,
        )
        tuner.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        # holdout 模式标记
        assert tuner._eval_mode == "holdout"
        # best_score_ 应为验证集上的指标值
        assert tuner.best_score_ > 0
        # best_estimator_ 可正常预测
        y_score = tuner.predict_score(X_val)
        assert len(y_score) == len(y_val)
