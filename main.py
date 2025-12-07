# main.py
from src.data_loader import load_and_merge_csv
from src.preprocess import clean_data, feature_engineering
from src.nlp_classifier import classify_industries
from src.eda import generate_basic_stats

DATA_PATH = f"D:\VisualStudioProjGuvi\IndustrialHumanVisuProjectMini\myenv\data"

if __name__ == "__main__":
    df = load_and_merge_csv(DATA_PATH)
    df = clean_data(df)
    df = feature_engineering(df)
    df = classify_industries(df)

    generate_basic_stats(df)

    print("\nRun `streamlit run src/app.py` to launch dashboard.")
