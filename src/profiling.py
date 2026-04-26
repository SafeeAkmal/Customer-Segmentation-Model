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

    profile["Persona"] = profile.apply(lambda row: assign_persona(row, profile), axis=1)
    return profile


def assign_persona(cluster_stats: pd.Series, profile: pd.DataFrame) -> str:
    max_spend_cluster = profile.loc[profile["TotalSpent"].idxmax(), "Cluster"]
    if cluster_stats["Cluster"] == max_spend_cluster:
        return "Premium Loyalist"
    return "Steady Multi-Channel"