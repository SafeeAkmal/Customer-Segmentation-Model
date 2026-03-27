import numpy as np
import pandas as pd
import pytest
from src.profiling import profile_clusters, assign_persona

def make_sample_df():
    """Minimal engineered DataFrame with all required columns."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'Age':               np.random.randint(25, 70, n),
        'Income':            np.random.randint(20000, 100000, n),
        'Family_Size':       np.random.randint(1, 5, n),
        'TotalSpent':        np.random.randint(0, 2000, n),
        'Recency':           np.random.randint(0, 99, n),
        'WebChannelShare':   np.random.uniform(0, 1, n),
        'CatalogChannelShare': np.random.uniform(0, 1, n),
        'StoreChannelShare': np.random.uniform(0, 1, n),
        'DealRate':          np.random.uniform(0, 1, n),
        'TotalPurchases':    np.random.randint(1, 30, n),
        'NumWebVisitsMonth': np.random.randint(0, 20, n),
    })

def test_profile_clusters_shape():
    """Profile must have one row per cluster."""
    df = make_sample_df()
    labels = np.array([0]*50 + [1]*50)
    profile = profile_clusters(df, labels)
    assert len(profile) == 2

def test_profile_has_persona_column():
    """Every cluster must be assigned a persona."""
    df = make_sample_df()
    labels = np.array([0]*50 + [1]*50)
    profile = profile_clusters(df, labels)
    assert 'Persona' in profile.columns
    assert profile['Persona'].isnull().sum() == 0

def test_persona_values_are_valid():
    """Persona must be one of the 6 defined values."""
    valid_personas = {
        'Premium Loyalist', 'Budget Conscious',
        'Digital Explorer', 'Recent High Value',
        'Disengaged', 'Steady Multi-Channel'
    }
    df = make_sample_df()
    labels = np.array([i % 4 for i in range(100)])
    profile = profile_clusters(df, labels)
    for persona in profile['Persona']:
        assert persona in valid_personas, f"Unknown persona: {persona}"

def test_assign_persona_premium():
    """High spend + high store share → Premium Loyalist."""
    stats = pd.Series({
        'TotalSpent': 2000, 'StoreShare': 0.6,
        'DealRate': 0.05, 'CatalogShare': 0.1,
        'WebShare': 0.2, 'NumWebVisitsMonth': 3,
        'Recency': 45
    })
    assert assign_persona(stats, high_spend_threshold=500) == 'Premium Loyalist'

def test_assign_persona_disengaged():
    """High recency + very low spend → Disengaged."""
    stats = pd.Series({
        'TotalSpent': 50, 'StoreShare': 0.3,
        'DealRate': 0.1, 'CatalogShare': 0.1,
        'WebShare': 0.2, 'NumWebVisitsMonth': 2,
        'Recency': 80
    })
    assert assign_persona(stats, high_spend_threshold=500) == 'Disengaged'

def test_count_column_sums_to_total():
    """Sum of Count across all clusters must equal total rows."""
    df = make_sample_df()
    labels = np.array([0]*40 + [1]*35 + [2]*25)
    profile = profile_clusters(df, labels)
    assert profile['Count'].sum() == 100