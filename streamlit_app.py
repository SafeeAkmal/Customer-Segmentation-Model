import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

from src.data_processing import engineer_features, transform_features
from src.profiling import assign_persona


@st.cache_resource
def load_artifacts():
    preprocess_path = Path("models/preprocessor.pkl")
    model_path = Path("models/best_cluster_model.pkl")

    if not preprocess_path.exists() or not model_path.exists():
        raise FileNotFoundError("Run run_training.py first to create models/preprocessor.pkl and models/best_cluster_model.pkl")

    pipeline = joblib.load(preprocess_path)
    model = joblib.load(model_path)
    return pipeline, model


def map_cluster_to_persona(cluster_value: int, profile_df: pd.DataFrame) -> str:
    if "Cluster" in profile_df.columns and "Persona" in profile_df.columns:
        match = profile_df[profile_df["Cluster"] == cluster_value]
        if not match.empty:
            return match.iloc[0]["Persona"]
    return "Unknown"


def main():
    st.title("Customer Segmentation Explorer")
    st.write("Unsupervised cluster assignment using trained model.")

    uploaded_file = st.file_uploader("Upload new customer CSV", type=["csv"])
    data_file = uploaded_file if uploaded_file is not None else "marketing_campaign.csv"

    df = pd.read_csv(data_file, parse_dates=["Dt_Customer"], dayfirst=True)
    st.sidebar.header("Sample options")
    if st.sidebar.button("Show data sample"):
        st.dataframe(df.head())

    pipeline, model = load_artifacts()

    if st.sidebar.checkbox("Use custom sample record"):
        data = {
            "Year_Birth": st.sidebar.number_input("Year_Birth", 1900, 2005, 1980),
            "Education": st.sidebar.selectbox("Education", ["Basic", "2n Cycle", "Graduation", "Master", "PhD"]),
            "Marital_Status": st.sidebar.selectbox("Marital_Status", ["Single", "Together", "Married", "Divorced", "Widow", "Alone", "Absurd", "YOLO"]),
            "Income": st.sidebar.number_input("Income", 0.0, 500000.0, 60000.0),
            "Kidhome": st.sidebar.slider("Kidhome", 0, 5, 0),
            "Teenhome": st.sidebar.slider("Teenhome", 0, 5, 0),
            "Recency": st.sidebar.slider("Recency", 0, 100, 30),
            "Dt_Customer": st.sidebar.date_input("Dt_Customer"),
            "MntWines": st.sidebar.number_input("MntWines", 0.0, 5000.0, 500.0),
            "MntFruits": st.sidebar.number_input("MntFruits", 0.0, 2000.0, 150.0),
            "MntMeatProducts": st.sidebar.number_input("MntMeatProducts", 0.0, 3000.0, 600.0),
            "MntFishProducts": st.sidebar.number_input("MntFishProducts", 0.0, 2000.0, 200.0),
            "MntSweetProducts": st.sidebar.number_input("MntSweetProducts", 0.0, 500.0, 60.0),
            "MntGoldProds": st.sidebar.number_input("MntGoldProds", 0.0, 500.0, 40.0),
            "NumWebPurchases": st.sidebar.slider("NumWebPurchases", 0, 50, 10),
            "NumCatalogPurchases": st.sidebar.slider("NumCatalogPurchases", 0, 50, 5),
            "NumStorePurchases": st.sidebar.slider("NumStorePurchases", 0, 50, 8),
            "NumWebVisitsMonth": st.sidebar.slider("NumWebVisitsMonth", 0, 50, 5),
            "NumDealsPurchases": st.sidebar.slider("NumDealsPurchases", 0, 30, 2),
            "Complain": st.sidebar.radio("Complain", [0, 1], index=0),
            "AcceptedCmp1": st.sidebar.radio("AcceptedCmp1", [0, 1], index=0),
            "AcceptedCmp2": st.sidebar.radio("AcceptedCmp2", [0, 1], index=0),
            "AcceptedCmp3": st.sidebar.radio("AcceptedCmp3", [0, 1], index=0),
            "AcceptedCmp4": st.sidebar.radio("AcceptedCmp4", [0, 1], index=0),
            "AcceptedCmp5": st.sidebar.radio("AcceptedCmp5", [0, 1], index=0),
        }

        single = pd.DataFrame([data])
        single = engineer_features(single)
        X_single = pipeline.transform(single)
        pred = model.predict(X_single)

        st.subheader("Single customer prediction")
        st.write("Cluster", int(pred[0]))

        profile_csv = Path("output/cluster_profile.csv")
        persona = "Unknown"
        if profile_csv.exists():
            profile_df = pd.read_csv(profile_csv)
            persona = map_cluster_to_persona(int(pred[0]), profile_df)
        st.write("Persona", persona)

    else:
        if st.sidebar.button("Score dataset"):
            df_eng = engineer_features(df)
            X = pipeline.transform(df_eng)
            labels = model.predict(X)
            df_eng["Cluster"] = labels
            st.dataframe(df_eng.groupby("Cluster").size().reset_index(name="Count"))

            st.write("### Example cluster crowns")
            st.dataframe(df_eng.groupby("Cluster").mean().reset_index().round(2))


if __name__ == "__main__":
    main()
