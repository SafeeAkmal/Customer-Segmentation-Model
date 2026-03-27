import os
import pytest

from src.data_processing import load_data, engineer_features
from run_training import run_all

DATA_PATH = 'data/raw/marketing_campaign.csv'
def test_load_data_has_rows():
    df = load_data(DATA_PATH)
    assert df.shape[0] > 0


def test_engineer_features_columns():
    df = load_data(DATA_PATH)
    df2 = engineer_features(df)
    assert "Age" in df2.columns
    assert "TotalSpent" in df2.columns


@pytest.mark.skip("Manual run; may take longer depending on environment")
def test_run_training_runs_without_error():
    run_all(DATA_PATH)
    assert os.path.exists("models/preprocessor.pkl")
    assert os.path.exists("models/best_cluster_model.pkl")
    assert os.path.exists("output/cluster_profile.csv")
