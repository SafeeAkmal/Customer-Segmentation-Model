import numpy as np
import pytest
from src.evaluation import evaluate_clustering

def test_perfect_clusters():
    """Two perfectly separated clusters should give high silhouette."""
    X = np.array([
        [0, 0], [0.1, 0], [0, 0.1],   # cluster 0
        [10, 10], [10.1, 10], [10, 10.1]  # cluster 1
    ])
    labels = np.array([0, 0, 0, 1, 1, 1])
    metrics = evaluate_clustering(X, labels)

    assert metrics["n_clusters"] == 2
    assert metrics["silhouette"] > 0.9      # near-perfect separation
    assert metrics["davies_bouldin"] < 0.1  # very low = good
    assert metrics["calinski_harabasz"] > 100

def test_noise_ignored_in_dbscan():
    """DBSCAN noise points (label=-1) must be excluded from metrics."""
    X = np.array([
        [0, 0], [0.1, 0],       # cluster 0
        [10, 10], [10.1, 10],   # cluster 1
        [99, 99]                # noise
    ])
    labels = np.array([0, 0, 1, 1, -1])
    metrics = evaluate_clustering(X, labels, ignore_noise=True)

    assert metrics["n_clusters"] == 2   # noise not counted
    assert not np.isnan(metrics["silhouette"])

def test_single_cluster_returns_nan():
    """Only 1 cluster — metrics cannot be computed, should return nan."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    labels = np.array([0, 0, 0])
    metrics = evaluate_clustering(X, labels)

    assert np.isnan(metrics["silhouette"])
    assert np.isnan(metrics["davies_bouldin"])

def test_all_noise_returns_nan():
    """All points labelled noise — should return nan gracefully."""
    X = np.array([[1, 2], [3, 4]])
    labels = np.array([-1, -1])
    metrics = evaluate_clustering(X, labels, ignore_noise=True)

    assert np.isnan(metrics["silhouette"])