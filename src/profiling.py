import pandas as pd


def profile_clusters(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    df_out = df.copy()
    df_out["Cluster"] = labels

    profile = (
        df_out.groupby("Cluster")
        .agg(
            Count=("Cluster", "size"),
            Avg_Age=("Age", "mean"),
            Median_Income=("Income", "median"),
            Avg_Family_Size=("Family_Size", "mean"),
            TotalSpent=("TotalSpent", "mean"),
            Recency=("Recency", "mean"),
            WebShare=("WebChannelShare", "mean"),
            CatalogShare=("CatalogChannelShare", "mean"),
            StoreShare=("StoreChannelShare", "mean"),
            DealRate=("DealRate", "mean"),
            TotalPurchases=("TotalPurchases", "mean"),
            NumWebVisitsMonth=("NumWebVisitsMonth", "mean"),
        )
        .reset_index()
    )

    high_spend_threshold = profile["TotalSpent"].quantile(0.7)
    profile["Persona"] = profile.apply(lambda row: assign_persona(row, high_spend_threshold), axis=1)
    return profile


def assign_persona(cluster_stats: pd.Series, high_spend_threshold: float) -> str:
    if cluster_stats["TotalSpent"] > high_spend_threshold and cluster_stats["StoreShare"] > 0.35:
        return "Premium Loyalist"
    if cluster_stats["DealRate"] > 0.2 and cluster_stats["CatalogShare"] > 0.25:
        return "Budget Conscious"
    if cluster_stats["WebShare"] > 0.5 and cluster_stats["NumWebVisitsMonth"] > 10:
        return "Digital Explorer"
    if cluster_stats["Recency"] < 30 and cluster_stats["TotalSpent"] > 500:
        return "Recent High Value"
    if cluster_stats["Recency"] > 60 and cluster_stats["TotalSpent"] < 200:
        return "Disengaged"

    return "Steady Multi-Channel"
