import os
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def load_data(csv_path: str) -> pd.DataFrame:
    # Read without parse_dates to avoid column space validation error
    df = pd.read_csv(csv_path, sep="\t", dayfirst=True, encoding="latin-1")
    # Strip leading/trailing spaces from column names (CSV has spaces in headers)
    df.columns = df.columns.str.strip()
    # Parse date column after stripping column names
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
    
    # Ensure numeric columns are properly typed (header spaces can interfere with inference)
    numeric_cols = [
        "Age", "Year_Birth", "Income", "Kidhome", "Teenhome", "Recency",
        "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds",
        "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth",
        "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5",
        "Complain", "Z_CostContact", "Z_Revenue", "Response", "ID"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with dirty Marital_Status values
    df = df[~df["Marital_Status"].isin(["Absurd", "YOLO", "Alone"])]

    # Remove extreme Income outlier (single known outlier at 666,666)
    df = df[df["Income"] < 600_000]
    df = df.copy()

    # Numeric demographic
    df["Age"] = (pd.Timestamp.now().year - df["Year_Birth"]).clip(lower=18, upper=90)
    df["Customer_For_Days"] = (pd.Timestamp.now() - df["Dt_Customer"]).dt.days
    df["Family_Size"] = df["Kidhome"] + df["Teenhome"] + \
    df["Marital_Status"].apply(lambda x: 2 if x in ["Married", "Together"] else 1)

    # Total spend and purchases
    spend_cols = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]
    df["TotalSpent"] = df[spend_cols].sum(axis=1)

    purchase_cols = [
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
        "NumDealsPurchases",
        "NumWebVisitsMonth",
    ]
    df["TotalPurchases"] = df[purchase_cols].sum(axis=1)

    df["SpendPerPurchase"] = np.where(df["TotalPurchases"] > 0, df["TotalSpent"] / df["TotalPurchases"], 0)
    df["DealRate"] = np.where(df["TotalPurchases"] > 0, df["NumDealsPurchases"] / df["TotalPurchases"], 0)

    # channel preference ratios
    total_channel = df["NumWebPurchases"] + df["NumCatalogPurchases"] + df["NumStorePurchases"]
    df["WebChannelShare"] = np.where(total_channel > 0, df["NumWebPurchases"] / total_channel, 0)
    df["CatalogChannelShare"] = np.where(total_channel > 0, df["NumCatalogPurchases"] / total_channel, 0)
    df["StoreChannelShare"] = np.where(total_channel > 0, df["NumStorePurchases"] / total_channel, 0)

    return df


def build_preprocessing_pipeline() -> Tuple[Pipeline, List[str]]:
    numeric_features = [
        "Age",
        "Income",
        "Kidhome",
        "Teenhome",
        "Recency",
        "Customer_For_Days",
        "TotalSpent",
        "TotalPurchases",
        "SpendPerPurchase",
        "DealRate",
        "NumWebVisitsMonth",
        "WebChannelShare",
        "CatalogChannelShare",
        "StoreChannelShare",
    ]

    categorical_features = ["Education", "Marital_Status"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    return pipeline, numeric_features, categorical_features


def prepare_features(csv_path: str) -> Tuple[pd.DataFrame, np.ndarray, Pipeline]:
    df = load_data(csv_path)
    df_eng = engineer_features(df)

    pipeline, numeric_features, categorical_features = build_preprocessing_pipeline()
    X = pipeline.fit_transform(df_eng)

    # derive categorical feature names after fit
    try:
        cat_names = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_features)
    except Exception:
        cat_names = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"].get_feature_names(categorical_features)

    feature_list = numeric_features + list(cat_names)
    X_df = pd.DataFrame(X, columns=feature_list, index=df_eng.index)

    return df_eng, X_df, pipeline


def transform_features(df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
    x = pipeline.transform(df)
    try:
        cat_names = pipeline.named_steps["preprocessor"] \
            .named_transformers_["cat"].named_steps["onehot"] \
            .get_feature_names_out(["Education", "Marital_Status"])
    except Exception:
        cat_names = pipeline.named_steps["preprocessor"] \
            .named_transformers_["cat"].named_steps["onehot"] \
            .get_feature_names(["Education", "Marital_Status"])

    numeric_names = [
        "Age", "Income", "Kidhome", "Teenhome", "Recency",
        "Customer_For_Days", "TotalSpent", "TotalPurchases",
        "SpendPerPurchase", "DealRate", "NumWebVisitsMonth",
        "WebChannelShare", "CatalogChannelShare", "StoreChannelShare"
    ]
    feature_names = numeric_names + list(cat_names)
    return pd.DataFrame(x, columns=feature_names, index=df.index)
