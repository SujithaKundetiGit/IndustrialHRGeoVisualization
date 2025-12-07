import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re

def normalize_columns(cols):
    """Same normalization function as preprocess.py"""
    new_cols = []
    for col in cols:
        col = col.lower().strip()
        col = re.sub(r"[\s\-]+", "_", col)  # spaces or hyphens → single underscore
        col = re.sub(r"_+", "_", col)       # collapse multiple underscores
        col = col.strip("_")
        new_cols.append(col)
    return new_cols

def classify_industries(df: pd.DataFrame, text_col="industry_name", n_clusters=10) -> pd.DataFrame:
    """
    Cluster industry sectors using NLP (TF-IDF + KMeans).
    Ensures text column exists and converts values to strings.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is None or empty in classify_industries()")

    df.columns = normalize_columns(df.columns)

    normalized_text_col = "_".join(text_col.lower().strip().split())
    normalized_text_col = re.sub(r"[\s\-]+", "_", normalized_text_col)
    normalized_text_col = re.sub(r"_+", "_", normalized_text_col).strip("_")

    if normalized_text_col not in df.columns:
        raise KeyError(
            f"Column '{text_col}' (normalized: '{normalized_text_col}') not found in classify_industries(). "
            f"Available columns: {df.columns.tolist()}"
        )

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(df[normalized_text_col].astype(str))  # ensure string

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["industry_cluster"] = kmeans.fit_predict(tfidf)

    return df
