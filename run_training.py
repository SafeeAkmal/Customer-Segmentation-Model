import json
import os
from pathlib import Path

import joblib

from src.data_processing import engineer_features, prepare_features
from src.clustering import run_kmeans, run_hierarchical, run_dbscan
from src.profiling import profile_clusters


def ensure_dirs():
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("output").mkdir(parents=True, exist_ok=True)


def select_best_model(results):
    # prioritize higher silhouette, then CH, then lower DB
    sorted_results = sorted(
        results,
        key=lambda r: (
            float("-inf") if r["metrics"]["silhouette"] is None or (r["metrics"]["silhouette"] != r["metrics"]["silhouette"]) else r["metrics"]["silhouette"],
            r["metrics"]["calinski_harabasz"],
            -r["metrics"]["davies_bouldin"],
        ),
        reverse=True,
    )
    return sorted_results[0]


def run_all(csv_path="marketing_campaign.csv"):
    ensure_dirs()

    df, X_df, pipeline = prepare_features(csv_path)
    X = X_df.values

    results = []

    for k in range(3, 7):
        model, labels, metrics = run_kmeans(X, n_clusters=k)
        results.append({"method": "kmeans", "k": k, "model": model, "labels": labels, "metrics": metrics})

    for k in range(3, 7):
        model, labels, metrics = run_hierarchical(X, n_clusters=k)
        results.append({"method": "hierarchical", "k": k, "model": model, "labels": labels, "metrics": metrics})

    for eps in [0.4, 0.5, 0.6, 0.7, 0.8]:
        model, labels, metrics = run_dbscan(X, eps=eps, min_samples=5)
        results.append({"method": "dbscan", "eps": eps, "model": model, "labels": labels, "metrics": metrics})

    best = select_best_model(results)
    print("Best model", best["method"], best.get("k", best.get("eps")), best["metrics"])

    joblib.dump(pipeline, "models/preprocessor.pkl")
    joblib.dump(best["model"], "models/best_cluster_model.pkl")

    df["Cluster"] = best["labels"]
    profile = profile_clusters(df, df["Cluster"])
    profile.to_csv("output/cluster_profile.csv", index=False)

    with open("models/selection_log.json", "w") as f:
        json.dump(
            {
                "best": {
                    "method": best["method"],
                    "params": {"k": best.get("k"), "eps": best.get("eps")},
                    "metrics": best["metrics"],
                },
                "candidates": [
                    {
                        "method": r["method"],
                        "params": {"k": r.get("k"), "eps": r.get("eps")},
                        "metrics": r["metrics"],
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )

    print("Saved pipeline, model, and profile.")


if __name__ == "__main__":
    run_all()
