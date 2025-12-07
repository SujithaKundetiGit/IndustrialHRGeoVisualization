import pandas as pd
import re

def normalize_columns(cols):
    """
    Normalize column names:
    - Lowercase
    - Strip spaces
    - Replace spaces/hyphens with underscores
    - Collapse multiple consecutive underscores
    - Strip leading/trailing underscores
    """
    new_cols = []
    for col in cols:
        col = col.lower().strip()
        col = re.sub(r"[\s\-]+", "_", col)  # spaces or hyphens → single underscore
        col = re.sub(r"_+", "_", col)       # collapse multiple underscores
        col = col.strip("_")
        new_cols.append(col)
    return new_cols

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values, remove duplicates, and normalize column names.
    Always returns a DataFrame.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is None or empty in clean_data()")

    df.columns = normalize_columns(df.columns)

    df.drop_duplicates(inplace=True)
    df.fillna(0, inplace=True)

    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features such as total workers.
    Raises KeyError if required columns are missing.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is None or empty in feature_engineering()")

    rename_map = {
        # Main Workers (Total)
        "main_workers_total_persons": "main_total_persons",
        "main_workers_total_males": "main_total_males",
        "main_workers_total_females": "main_total_females",

        # Main Workers (Rural)
        "main_workers_rural_persons": "main_rural_persons",
        "main_workers_rural_males": "main_rural_males",
        "main_workers_rural_females": "main_rural_females",

        # Main Workers (Urban)
        "main_workers_urban_persons": "main_urban_persons",
        "main_workers_urban_males": "main_urban_males",
        "main_workers_urban_females": "main_urban_females",
    }

    df = df.rename(columns=rename_map)

    required_cols = ["main_total_males", "main_total_females"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(
                f"Required column '{col}' missing in feature_engineering(). "
                f"Available columns: {df.columns.tolist()}"
            )

    # Compute total workers
    df["total_workers"] = df["main_total_males"] + df["main_total_females"]

    # Optional: compute rural percentage safely
    df["rural_percentage"] = df.get("main_rural_persons", 0) / df["main_total_persons"].replace(0, 1)

    return df
