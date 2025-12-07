import streamlit as st
import pandas as pd
import os
from preprocess import clean_data, feature_engineering
from nlp_classifier import classify_industries
import plotly.express as px

# ------------------------
# DATA LOADING
# ------------------------
DATA_PATH = r"D:\VisualStudioProjGuvi\IndustrialHumanVisuProjectMini\myenv\data"

def load_and_merge_csv(data_dir):
    dfs = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(data_dir, filename)
            # Attempt to read with common encodings
            try:
                df = pd.read_csv(file_path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding="cp1252")
            dfs.append(df)
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        st.success(f"Loaded {len(merged_df)} rows from CSV files.")
        return merged_df
    else:
        st.warning("No CSV files found in the data directory.")
        return pd.DataFrame()


# ------------------------
# VISUALIZATION FUNCTIONS
# ------------------------
def plot_workers_by_state(df):
    if "state_code" not in df.columns:
        st.warning("Column 'state_code' missing in the dataset for plotting.")
        return {}
    state_totals = df.groupby("state_code")["total_workers"].sum().reset_index()
    fig = px.bar(state_totals, x="state_code", y="total_workers",
                 title="Total Workers by State",
                 labels={"state_code": "State", "total_workers": "Total Workers"})
    return fig

def plot_industry_distribution(df):
    if "industry_cluster" not in df.columns:
        st.warning("Industry clusters not computed.")
        return {}
    industry_totals = df.groupby("industry_cluster")["total_workers"].sum().reset_index()
    fig = px.pie(industry_totals, values="total_workers", names="industry_cluster",
                 title="Industry Cluster Distribution")
    return fig

# ------------------------
# MAIN DASHBOARD
# ------------------------
def main():
    st.title("📊 Industrial Human Resource Geo-Visualization Dashboard")

    df = load_and_merge_csv(DATA_PATH)
    if df.empty:
        st.stop()

    # ------------------------
    # PREPROCESSING
    # ------------------------
    try:
        df = clean_data(df)
        df = feature_engineering(df)
    except Exception as e:
        st.error(f"Feature engineering failed: {e}")
        st.stop()

    # ------------------------
    # NLP CLUSTERING
    # ------------------------
    try:
        # Use 'nic_name' as the industry column
        df = classify_industries(df, text_col="nic_name", n_clusters=10)
    except Exception as e:
        st.error(f"Industry classification failed: {e}")
        st.stop()

    # ------------------------
    # DATA PREVIEW
    # ------------------------
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ------------------------
    # WORKERS BY STATE
    # ------------------------
    st.subheader("Workers by State")
    fig_state = plot_workers_by_state(df)
    if fig_state:
        st.plotly_chart(fig_state)

    # ------------------------
    # INDUSTRY CLUSTER OVERVIEW
    # ------------------------
    st.subheader("Industry Cluster Overview")
    fig_industry = plot_industry_distribution(df)
    if fig_industry:
        st.plotly_chart(fig_industry)

    # ------------------------
    # INTERACTIVE STATE FILTER
    # ------------------------
    normalized_state_col = "state_code"
    if normalized_state_col in df.columns:
        state = st.selectbox("Select State", df[normalized_state_col].unique())
        st.subheader(f"Data for {state}")
        st.dataframe(df[df[normalized_state_col] == state])
    else:
        st.warning("Column 'state_code' not found. Cannot filter by state.")


if __name__ == "__main__":
    main()
