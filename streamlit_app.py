import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data_processing import load_data, engineer_features

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f2547 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover { transform: translateY(-2px); border-color: #3b82f6; }
    .metric-card .label {
        font-size: 11px; font-weight: 600; letter-spacing: 0.12em;
        text-transform: uppercase; color: #64748b; margin-bottom: 8px;
    }
    .metric-card .value { font-size: 28px; font-weight: 700; color: #f8fafc; line-height: 1; }
    .metric-card .sub { font-size: 12px; color: #64748b; margin-top: 4px; }
    .section-title {
        font-size: 13px; font-weight: 600; letter-spacing: 0.12em;
        text-transform: uppercase; color: #475569;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 8px; margin-bottom: 16px;
    }
    .persona-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .persona-card h4 { color: #93c5fd; margin: 0 0 8px 0; font-size: 15px; }
    .persona-card p  { color: #94a3b8; margin: 0; font-size: 13px; line-height: 1.5; }
    .rec-item {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
        color: #cbd5e1;
        font-size: 13px;
    }
    .rec-item::before { content: "→  "; color: #3b82f6; font-weight: 700; }
    .finding-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    .finding-box p { color: #94a3b8; margin: 0; font-size: 13px; line-height: 1.6; }
    .finding-box strong { color: #fbbf24; }
    .stButton>button {
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; font-size: 13px; padding: 10px 20px;
        transition: opacity 0.2s;
    }
    .stButton>button:hover { opacity: 0.88; }
    h1, h2, h3 { color: #f8fafc !important; }
    .stTabs [data-baseweb="tab"] { color: #64748b; font-weight: 500; }
    .stTabs [aria-selected="true"] { color: #3b82f6 !important; }
</style>
""", unsafe_allow_html=True)

DATA_PATH     = ROOT / "data" / "raw" / "marketing_campaign.csv"
PIPELINE_PATH = ROOT / "models" / "pipeline.pkl"
MODEL_PATH    = ROOT / "models" / "best_model.pkl"
PROFILE_PATH  = ROOT / "models" / "cluster_profiles.csv"

PERSONA_DESCRIPTIONS = {
    "Premium Loyalist": (
        "High-income, high-spend customers who are the most valuable segment. "
        "They spend an average of 7x more than the other segment across all product categories, "
        "with particular concentration in wines and meat products. They have higher catalog channel "
        "usage and respond well to premium, curated offers."
    ),
    "Steady Multi-Channel": (
        "The larger, broader customer group with moderate-to-low income and significantly lower "
        "spend across all categories. They are more web-active and price-aware, making them "
        "responsive to digital campaigns and value-based promotions. This group has the most "
        "room for growth through targeted upsell strategies."
    ),
}

PERSONA_RECOMMENDATIONS = {
    "Premium Loyalist": [
        "Enrol in an exclusive loyalty programme with priority access and early product releases.",
        "Offer premium bundle deals on wines and meat products — their dominant spend categories.",
        "Use personalised catalog mailings — they over-index on catalog purchasing vs the other segment.",
        "Invite to VIP in-store events and tastings to deepen brand connection.",
        "Protect this segment aggressively — churn from even a fraction of these customers has outsized revenue impact.",
    ],
    "Steady Multi-Channel": [
        "Prioritise digital channels — this segment has higher web visit frequency than Premium Loyalists.",
        "Lead with value messaging: bundle offers, loyalty points, and introductory discounts.",
        "Use cross-sell campaigns targeting their most-purchased low-spend categories (fruits, sweets).",
        "Identify the top 20% of this segment by TotalSpent and run targeted upsell to migrate them toward Premium Loyalist behaviour.",
        "A/B test web vs in-store promotions — this segment uses both channels at a more balanced rate.",
    ],
}

PLOTLY_THEME = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor":  "rgba(0,0,0,0)",
    "font":          {"color": "#94a3b8"},
}
GRID_COLOR = "#1e293b"


@st.cache_resource(show_spinner="Loading models...")
def load_artifacts():
    missing = [str(p) for p in [PIPELINE_PATH, MODEL_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Model files not found: {missing}\n"
            "Run 03_Modeling.ipynb and 04_Evaluation.ipynb first."
        )
    return joblib.load(PIPELINE_PATH), joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Loading profiles...")
def load_profiles():
    if not PROFILE_PATH.exists():
        raise FileNotFoundError(f"{PROFILE_PATH} not found. Run 04_Evaluation.ipynb first.")
    return pd.read_csv(PROFILE_PATH)


@st.cache_data(show_spinner="Loading dataset...")
def load_dataset():
    if not DATA_PATH.exists():
        return None
    return load_data(str(DATA_PATH))


def predict_customer(input_dict: dict, pipeline, model, profiles: pd.DataFrame):
    defaults = {
        "AcceptedCmp1": 0, "AcceptedCmp2": 0, "AcceptedCmp3": 0,
        "AcceptedCmp4": 0, "AcceptedCmp5": 0, "Response": 0,
        "Complain": 0, "Z_CostContact": 3, "Z_Revenue": 11, "ID": 0,
    }
    for k, v in defaults.items():
        if k not in input_dict:
            input_dict[k] = v

    if not isinstance(input_dict["Dt_Customer"], pd.Timestamp):
        input_dict["Dt_Customer"] = pd.Timestamp(input_dict["Dt_Customer"])

    df_single   = pd.DataFrame([input_dict])
    df_eng      = engineer_features(df_single)
    X           = pipeline.transform(df_eng)
    cluster     = int(model.predict(X)[0])
    match       = profiles[profiles["Cluster"] == cluster]
    persona     = match.iloc[0]["Persona"] if not match.empty else "Unknown"
    profile_row = match.iloc[0] if not match.empty else None

    return cluster, persona, profile_row


def main():
    try:
        pipeline, model = load_artifacts()
        profiles        = load_profiles()
    except FileNotFoundError as e:
        st.error(f"**Setup required:** {e}")
        st.stop()

    st.markdown("## Customer Personality Segmentation")
    st.markdown(
        "<span style='color:#64748b;font-size:14px;'>"
        "Unsupervised clustering · K-Means · Hierarchical · DBSCAN · "
        "Customer Personality Analysis Dataset"
        "</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "Predict Single Customer",
        "Dataset Overview",
        "Segment Profiles",
    ])

    with tab1:
        st.markdown("### Enter Customer Attributes")
        st.caption("Adjust the values below to predict which of the two natural segments this customer belongs to.")

        st.markdown(
            '<div class="finding-box"><p>'
            '<strong>Key Finding:</strong> All three clustering algorithms (K-Means, Hierarchical, DBSCAN) '
            'independently confirm <strong>2 natural segments</strong> in this dataset. The Silhouette Score '
            'peaks at k=2 (0.2298) and declines monotonically for all higher k values. The dominant divide '
            'is income and total spend: Premium Loyalists vs Steady Multi-Channel customers.'
            '</p></div>',
            unsafe_allow_html=True,
        )

        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("**Demographics**")
                year_birth  = st.number_input("Birth Year", 1940, 2000, 1975)
                marital     = st.selectbox("Marital Status",
                                ["Married", "Together", "Single", "Divorced", "Widow"])
                education   = st.selectbox("Education",
                                ["Graduation", "PhD", "Master", "2n Cycle", "Basic"])
                income      = st.number_input("Annual Income ($)", 0, 150000, 52000, step=500)
                kidhome     = st.selectbox("Children at Home",  [0, 1, 2])
                teenhome    = st.selectbox("Teenagers at Home", [0, 1, 2])
                recency     = st.slider("Days Since Last Purchase", 0, 99, 30)
                dt_customer = st.date_input("Enrolment Date", value=pd.Timestamp("2013-06-15"))

            with c2:
                st.markdown("**Spending ($)**")
                mnt_wines  = st.number_input("Wines",          0, 1500, 200)
                mnt_fruits = st.number_input("Fruits",         0, 200,  10)
                mnt_meat   = st.number_input("Meat Products",  0, 1750, 100)
                mnt_fish   = st.number_input("Fish Products",  0, 300,  20)
                mnt_sweet  = st.number_input("Sweet Products", 0, 300,  10)
                mnt_gold   = st.number_input("Gold Products",  0, 400,  30)

            with c3:
                st.markdown("**Purchase Behaviour**")
                num_web     = st.slider("Web Purchases",     0, 30, 4)
                num_catalog = st.slider("Catalog Purchases", 0, 30, 2)
                num_store   = st.slider("Store Purchases",   0, 30, 6)
                num_deals   = st.slider("Deal Purchases",    0, 15, 2)
                num_web_vis = st.slider("Web Visits/Month",  0, 20, 5)

            submitted = st.form_submit_button("Predict Segment", use_container_width=True)

        if submitted:
            input_dict = {
                "Year_Birth": year_birth, "Education": education,
                "Marital_Status": marital, "Income": income,
                "Kidhome": kidhome, "Teenhome": teenhome,
                "Recency": recency, "Dt_Customer": pd.Timestamp(str(dt_customer)),
                "MntWines": mnt_wines, "MntFruits": mnt_fruits,
                "MntMeatProducts": mnt_meat, "MntFishProducts": mnt_fish,
                "MntSweetProducts": mnt_sweet, "MntGoldProds": mnt_gold,
                "NumWebPurchases": num_web, "NumCatalogPurchases": num_catalog,
                "NumStorePurchases": num_store, "NumDealsPurchases": num_deals,
                "NumWebVisitsMonth": num_web_vis,
            }

            cluster, persona, profile_row = predict_customer(input_dict, pipeline, model, profiles)

            total_spent = mnt_wines + mnt_fruits + mnt_meat + mnt_fish + mnt_sweet + mnt_gold
            age         = 2024 - year_birth
            family_size = kidhome + teenhome + (2 if marital in ["Married", "Together"] else 1)

            m1, m2, m3, m4 = st.columns(4)
            for col, label, value, sub in [
                (m1, "Predicted Cluster", f"Cluster {cluster}", ""),
                (m2, "Persona",           persona,              ""),
                (m3, "Total Spend",       f"${total_spent:,}",  "entered above"),
                (m4, "Age / Family",      f"{age}y  |  {family_size}", "age | household size"),
            ]:
                col.markdown(
                    f'<div class="metric-card">'
                    f'<div class="label">{label}</div>'
                    f'<div class="value">{value}</div>'
                    f'<div class="sub">{sub}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("")
            left, right = st.columns(2)

            with left:
                st.markdown('<div class="section-title">Persona Description</div>', unsafe_allow_html=True)
                desc = PERSONA_DESCRIPTIONS.get(persona, "No description available.")
                st.markdown(
                    f'<div class="persona-card"><h4>{persona}</h4><p>{desc}</p></div>',
                    unsafe_allow_html=True,
                )

                if profile_row is not None:
                    st.markdown(
                        '<div class="section-title" style="margin-top:20px">Segment Average Profile</div>',
                        unsafe_allow_html=True,
                    )
                    profile_display = {
                        "Avg Age":       f"{profile_row.get('Avg_Age', 0):.0f}",
                        "Median Income": f"${profile_row.get('Median_Income', 0):,.0f}",
                        "Avg Spend":     f"${profile_row.get('TotalSpent', 0):,.0f}",
                        "Avg Recency":   f"{profile_row.get('Recency', 0):.0f} days",
                        "Deal Rate":     f"{profile_row.get('DealRate', 0):.1%}",
                        "Segment Size":  f"{profile_row.get('Count', 0):,} customers",
                    }
                    for k, v in profile_display.items():
                        st.markdown(
                            f'<div style="display:flex;justify-content:space-between;'
                            f'padding:8px 0;border-bottom:1px solid #1e293b;">'
                            f'<span style="color:#64748b;font-size:13px">{k}</span>'
                            f'<span style="color:#e2e8f0;font-weight:600;font-size:13px">{v}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            with right:
                st.markdown('<div class="section-title">Marketing Recommendations</div>', unsafe_allow_html=True)
                for rec in PERSONA_RECOMMENDATIONS.get(persona, ["No recommendations available."]):
                    st.markdown(f'<div class="rec-item">{rec}</div>', unsafe_allow_html=True)

                if profile_row is not None:
                    st.markdown(
                        '<div class="section-title" style="margin-top:20px">Channel Preference</div>',
                        unsafe_allow_html=True,
                    )
                    channels = {
                        "Web":     profile_row.get("WebShare",     0),
                        "Catalog": profile_row.get("CatalogShare", 0),
                        "Store":   profile_row.get("StoreShare",   0),
                    }
                    fig = go.Figure(go.Bar(
                        x=list(channels.keys()),
                        y=[v * 100 for v in channels.values()],
                        marker_color=["#3b82f6", "#8b5cf6", "#10b981"],
                        text=[f"{v:.0%}" for v in channels.values()],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        height=220, margin=dict(t=10, b=10, l=0, r=0),
                        yaxis_title="Share (%)", showlegend=False, **PLOTLY_THEME,
                    )
                    fig.update_xaxes(showgrid=False)
                    fig.update_yaxes(gridcolor=GRID_COLOR)
                    st.plotly_chart(fig, use_container_width=True)

    with tab2:
        df_raw = load_dataset()

        if df_raw is None:
            st.warning("Raw dataset not found at `data/raw/marketing_campaign.csv`.")
        else:
            st.markdown("### Dataset at a Glance")
            g1, g2, g3, g4 = st.columns(4)
            for col, label, value in [
                (g1, "Total Customers",     f"{len(df_raw):,}"),
                (g2, "Raw Features",        "29"),
                (g3, "Engineered Features", "10"),
                (g4, "Natural Segments",    "2"),
            ]:
                col.markdown(
                    f'<div class="metric-card">'
                    f'<div class="label">{label}</div>'
                    f'<div class="value">{value}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("")
            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("**Income Distribution**")
                df_clean = df_raw[df_raw["Income"] < 200_000].dropna(subset=["Income"])
                fig = px.histogram(df_clean, x="Income", nbins=50,
                                   color_discrete_sequence=["#3b82f6"])
                fig.update_layout(height=300, margin=dict(t=10, b=10, l=0, r=0),
                                  showlegend=False, **PLOTLY_THEME)
                fig.update_xaxes(title="Income ($)", showgrid=False)
                fig.update_yaxes(title="Count", gridcolor=GRID_COLOR)
                st.plotly_chart(fig, use_container_width=True)

            with col_right:
                st.markdown("**Education Distribution**")
                edu_counts = df_raw["Education"].str.strip().value_counts().reset_index()
                edu_counts.columns = ["Education", "Count"]
                fig = px.bar(edu_counts, x="Count", y="Education",
                             orientation="h", color_discrete_sequence=["#8b5cf6"])
                fig.update_layout(height=300, margin=dict(t=10, b=10, l=0, r=0),
                                  showlegend=False, **PLOTLY_THEME)
                fig.update_xaxes(gridcolor=GRID_COLOR)
                fig.update_yaxes(showgrid=False)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Data Sample**")
            st.dataframe(df_raw.head(10), use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("### Segment Profiles")
        st.markdown(
            '<div class="finding-box"><p>'
            '<strong>Why 2 segments?</strong> K-Means Silhouette Score peaks at k=2 (0.2298) and '
            'declines monotonically for k=3 through k=12. Hierarchical clustering dendrogram shows '
            'two primary branches at the highest distance threshold. DBSCAN labels over 75% of points '
            'as noise across all parameter combinations tested. All three algorithms independently '
            'confirm the same conclusion: one dominant binary split between high-value and standard customers.'
            '</p></div>',
            unsafe_allow_html=True,
        )

        for _, row in profiles.iterrows():
            persona    = row.get("Persona", "Unknown")
            cluster_id = int(row.get("Cluster", 0))
            count      = int(row.get("Count", 0))
            spend      = row.get("TotalSpent", 0)
            income     = row.get("Median_Income", 0)
            age        = row.get("Avg_Age", 0)

            with st.expander(f"Cluster {cluster_id} — {persona}  ({count:,} customers)", expanded=True):
                e1, e2, e3 = st.columns(3)
                e1.metric("Avg Spend",     f"${spend:,.0f}")
                e2.metric("Median Income", f"${income:,.0f}")
                e3.metric("Avg Age",       f"{age:.0f} yrs")

                st.markdown(
                    f'<div class="persona-card">'
                    f'<h4>{persona}</h4>'
                    f'<p>{PERSONA_DESCRIPTIONS.get(persona, "")}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                st.markdown("**Recommendations**")
                for rec in PERSONA_RECOMMENDATIONS.get(persona, []):
                    st.markdown(f'<div class="rec-item">{rec}</div>', unsafe_allow_html=True)

        st.markdown("")
        st.markdown("**Full Profile Table**")
        display_cols = [c for c in [
            "Cluster", "Persona", "Count", "Avg_Age", "Median_Income",
            "TotalSpent", "Recency", "DealRate", "WebShare", "CatalogShare", "StoreShare",
        ] if c in profiles.columns]
        st.dataframe(profiles[display_cols].round(3), use_container_width=True, hide_index=True)

        st.markdown("**Spend & Income Comparison**")
        x_axis = profiles["Persona"] if "Persona" in profiles.columns else profiles["Cluster"].astype(str)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Avg Spend",     x=x_axis, y=profiles["TotalSpent"],    marker_color="#3b82f6"))
        fig.add_trace(go.Bar(name="Median Income", x=x_axis, y=profiles["Median_Income"], marker_color="#10b981"))
        fig.update_layout(
            barmode="group", height=350,
            margin=dict(t=10, b=40, l=0, r=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            **PLOTLY_THEME,
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor=GRID_COLOR, title="$")
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()