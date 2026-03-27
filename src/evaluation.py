from typing import Dict

import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def evaluate_clustering(X: np.ndarray, labels: np.ndarray, ignore_noise: bool = False) -> Dict[str, float]:
    metrics = {
        "n_clusters": int(len(set(labels) - {-1})) if ignore_noise else int(len(set(labels))),
        "silhouette": float("nan"),
        "davies_bouldin": float("nan"),
        "calinski_harabasz": float("nan"),
    }

    # valid labels for silhouette/higher metrics: at least 2 clusters
    usable_labels = labels
    if ignore_noise:
        valid_mask = labels != -1
        if valid_mask.sum() < 2:
            return metrics
        usable_labels = labels[valid_mask]
        usable_X = X[valid_mask]
    else:
        usable_X = X

    if len(set(usable_labels)) >= 2:
        metrics["silhouette"] = silhouette_score(usable_X, usable_labels)
        metrics["davies_bouldin"] = davies_bouldin_score(usable_X, usable_labels)
        metrics["calinski_harabasz"] = calinski_harabasz_score(usable_X, usable_labels)
    return metrics
