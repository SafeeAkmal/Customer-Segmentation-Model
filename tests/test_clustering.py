import numpy as np
import pytest
from sklearn.datasets import make_blobs
from src.clustering import (
    find_optimal_k, run_kmeans,
    run_hierarchical, run_dbscan,
    dbscan_grid_search, save_model, load_model
)

# Synthetic well-separated data for all tests
X_BLOBS, _ = make_blobs(n_samples=300, centers=4,
                         cluster_std=0.5, random_state=42)

def test_find_optimal_k_returns_dataframe():
    results = find_optimal_k(X_BLOBS, k_range=range(2, 6))
    assert len(results) == 4
    assert 'silhouette' in results.columns
    assert 'inertia' in results.columns

def test_find_optimal_k_selects_4():
    """With 4 well-separated blobs, optimal k should be 4."""
    results = find_optimal_k(X_BLOBS, k_range=range(2, 7))
    best_k = results.loc[results['silhouette'].idxmax(), 'k']
    assert best_k == 4, f"Expected k=4, got k={best_k}"

def test_run_kmeans_label_count():
    """KMeans labels must have exactly n_clusters unique values."""
    _, labels, metrics = run_kmeans(X_BLOBS, n_clusters=4)
    assert len(set(labels)) == 4
    assert metrics['silhouette'] > 0.5

def test_run_hierarchical_label_count():
    """Hierarchical labels must have exactly n_clusters unique values."""
    _, labels, metrics = run_hierarchical(X_BLOBS, n_clusters=4)
    assert len(set(labels)) == 4
    assert metrics['silhouette'] > 0.5

def test_run_dbscan_finds_clusters():
    """DBSCAN on clean blobs should find clusters with low noise."""
    _, labels, metrics = run_dbscan(X_BLOBS, eps=0.8, min_samples=5)
    noise_pct = (labels == -1).sum() / len(labels)
    assert noise_pct < 0.15, f"Too much noise: {noise_pct:.2%}"
    assert metrics['n_clusters'] >= 2

def test_dbscan_grid_search_shape():
    """Grid search must return eps × min_samples rows."""
    eps_vals = [0.5, 1.0]
    ms_vals  = [3, 5]
    results  = dbscan_grid_search(X_BLOBS, eps_vals, ms_vals)
    assert len(results) == 4   # 2 × 2
    assert 'noise_pct' in results.columns

def test_save_and_load_model(tmp_path):
    """Model saved with save_model must be loadable and predict correctly."""
    model, labels, _ = run_kmeans(X_BLOBS, n_clusters=4)
    path = str(tmp_path / 'kmeans_test.pkl')
    save_model(model, path)
    loaded = load_model(path)
    new_labels = loaded.predict(X_BLOBS)
    # Loaded model must produce same number of unique clusters
    assert len(set(new_labels)) == 4