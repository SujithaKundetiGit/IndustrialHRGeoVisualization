# src/visualization.py
import plotly.express as px
import pandas as pd

def plot_workers_by_state(df: pd.DataFrame):
    fig = px.choropleth(
        df,
        locations="state",
        locationmode="india states",
        color="total_workers",
        title="Workers by State",
        scope="asia"
    )
    return fig

def plot_industry_distribution(df: pd.DataFrame):
    fig = px.bar(
        df.groupby("industry_cluster")["total_workers"].sum().reset_index(),
        x="industry_cluster",
        y="total_workers",
        title="Industry Cluster Distribution"
    )
    return fig
