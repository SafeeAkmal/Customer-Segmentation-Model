import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.base import BaseEstimator
from typing import Any, Dict, Tuple
try:
    from .evaluation import evaluate_clustering
except ImportError:
    from evaluation import evaluate_clustering
    
def find_optimal_k(X: np.ndarray, k_range: range = range(2, 13)) -> pd.DataFrame:
    results = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        metrics = evaluate_clustering(X, labels)
        results.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": metrics["silhouette"],
            "davies_bouldin": metrics["davies_bouldin"],
            "calinski_harabasz": metrics["calinski_harabasz"],
        })
    return pd.DataFrame(results)


def run_kmeans(X: np.ndarray, n_clusters: int = 2, random_state: int = 42) -> Tuple[KMeans, np.ndarray, Dict[str, float]]:
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    metrics = evaluate_clustering(X, labels)
    return model, labels, metrics


def run_hierarchical(X: np.ndarray, n_clusters: int = 2, linkage: str = "ward") -> Tuple[AgglomerativeClustering, np.ndarray, Dict[str, float]]:
    model = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage=linkage)
    labels = model.fit_predict(X)
    metrics = evaluate_clustering(X, labels)
    return model, labels, metrics


def dbscan_grid_search(
    X: np.ndarray,
    eps_values: list = [0.3, 0.5, 0.8, 1.0, 1.5],
    min_samples_values: list = [3, 5, 7, 10],
) -> pd.DataFrame:
    results = []
    for eps in eps_values:
        for ms in min_samples_values:
            model = DBSCAN(eps=eps, min_samples=ms)
            labels = model.fit_predict(X)
            n_noise = int((labels == -1).sum())
            noise_pct = round(n_noise / len(labels) * 100, 2)
            metrics = evaluate_clustering(X, labels, ignore_noise=True)
            results.append({
                "eps": eps,
                "min_samples": ms,
                "n_clusters": metrics["n_clusters"],
                "noise_pct": noise_pct,
                "silhouette": metrics["silhouette"],
            })
    return pd.DataFrame(results)


def run_dbscan(X: np.ndarray, eps: float = 0.6, min_samples: int = 5) -> Tuple[DBSCAN, np.ndarray, Dict[str, float]]:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    metrics = evaluate_clustering(X, labels, ignore_noise=True)
    return model, labels, metrics


def save_model(model: BaseEstimator, save_path: str) -> None:
    import joblib
    joblib.dump(model, save_path)


def load_model(load_path: str) -> Any:
    import joblib
    return joblib.load(load_path)