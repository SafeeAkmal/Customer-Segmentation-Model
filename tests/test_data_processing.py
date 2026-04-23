import numpy as np
import pandas as pd
import pytest
from src.data_processing import load_data, engineer_features, prepare_features

DATA_PATH = 'data/raw/marketing_campaign.csv'

# ── load_data ─────────────────────────────────────────────────────────────────

def test_load_data_shape():
    """Raw dataset should have 2240 rows and 29 columns."""
    df = load_data(DATA_PATH)
    assert df.shape[0] == 2240
    assert df.shape[1] == 29

def test_load_data_dt_customer_is_datetime():
    """Dt_Customer must be parsed as datetime, not string."""
    df = load_data(DATA_PATH)
    assert pd.api.types.is_datetime64_any_dtype(df['Dt_Customer']), \
        "Dt_Customer should be datetime — check parse_dates in load_data()"

def test_load_data_no_extra_columns():
    """Tab separator must be correct — wrong sep collapses all cols into one."""
    df = load_data(DATA_PATH)
    assert 'Income' in df.columns
    assert 'MntWines' in df.columns
    assert 'Marital_Status' in df.columns

# ── engineer_features ─────────────────────────────────────────────────────────

def test_dirty_marital_status_removed():
    """Rows with Absurd, YOLO, Alone must be dropped."""
    df = load_data(DATA_PATH)
    df_eng = engineer_features(df)
    dirty = df_eng['Marital_Status'].isin(['Absurd', 'YOLO', 'Alone'])
    assert dirty.sum() == 0, f"Found {dirty.sum()} dirty Marital_Status rows"

def test_cleaned_shape():
    df = load_data(DATA_PATH)
    df_eng = engineer_features(df)
    assert df_eng.shape[0] == 2208   # 2240 - 24 nulls dropped? No — 
    # Actually: 2240 raw - 1 income outlier - 7 dirty marital = 2232
    # But nulls are imputed not dropped, so expect 2232
    assert df_eng.shape[0] == 2232
    
def test_income_outlier_removed():
    """Income outlier at 666,666 must be removed."""
    df = load_data(DATA_PATH)
    df_eng = engineer_features(df)
    assert df_eng['Income'].max() < 600_000

def test_engineered_columns_exist():
    """All 10 engineered columns must be present after engineering."""
    df = load_data(DATA_PATH)
    df_eng = engineer_features(df)
    expected = [
        'Age', 'Customer_For_Days', 'Family_Size',
        'TotalSpent', 'TotalPurchases',
        'SpendPerPurchase', 'DealRate',
        'WebChannelShare', 'CatalogChannelShare', 'StoreChannelShare'
    ]
    for col in expected:
        assert col in df_eng.columns, f"Missing engineered column: {col}"

def test_age_clipped():
    """Age must be between 18 and 90 — no implausible birth years."""
    df = load_data(DATA_PATH)
    df_eng = engineer_features(df)
    assert df_eng['Age'].min() >= 18
    assert df_eng['Age'].max() <= 90

def test_no_negative_values_in_ratios():
    """Channel shares and DealRate must be between 0 and 1."""
    df = load_data(DATA_PATH)
    df_eng = engineer_features(df)
    for col in ['WebChannelShare', 'CatalogChannelShare',
                'StoreChannelShare', 'DealRate']:
        assert df_eng[col].min() >= 0.0, f"{col} has negative values"
        assert df_eng[col].max() <= 1.0, f"{col} exceeds 1.0"

def test_total_spent_equals_sum_of_mnt():
    """TotalSpent must equal the exact sum of all 6 Mnt* columns."""
    df = load_data(DATA_PATH)
    df_eng = engineer_features(df)
    mnt_cols = ['MntWines','MntFruits','MntMeatProducts',
                'MntFishProducts','MntSweetProducts','MntGoldProds']
    expected = df_eng[mnt_cols].sum(axis=1)
    pd.testing.assert_series_equal(df_eng['TotalSpent'], expected,
                                   check_names=False)

def test_family_size_is_positive():
    """Family_Size must be at least 1 for every customer."""
    df = load_data(DATA_PATH)
    df_eng = engineer_features(df)
    assert df_eng['Family_Size'].min() >= 1

# ── prepare_features (full pipeline) ─────────────────────────────────────────

def test_prepare_features_no_nulls():
    """Scaled feature matrix must have zero nulls after pipeline."""
    df_eng, X_df, pipeline = prepare_features(DATA_PATH)
    assert X_df.isnull().sum().sum() == 0, "Nulls remain after pipeline"

def test_prepare_features_scaled():
    """Numeric features should be approximately zero-mean after scaling."""
    df_eng, X_df, pipeline = prepare_features(DATA_PATH)
    numeric_cols = ['Age', 'Income', 'TotalSpent', 'DealRate']
    for col in numeric_cols:
        mean = X_df[col].mean()
        assert abs(mean) < 0.1, f"{col} mean is {mean:.4f} — scaling may be wrong"

def test_response_not_in_features():
    """Response must never appear in the scaled feature matrix."""
    df_eng, X_df, pipeline = prepare_features(DATA_PATH)
    assert 'Response' not in X_df.columns, \
        "Response column leaked into feature matrix — remove it from pipeline"