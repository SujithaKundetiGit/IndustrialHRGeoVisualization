import pandas as pd

def generate_basic_stats(df: pd.DataFrame):
    """
    Generate basic information about the dataset.
    - Column info
    - Descriptive statistics
    - Top industries by total workers
    - Top states by total workers
    """

    # Normalize column names to ensure consistency
    df.columns = [col.strip().lower().replace(" ", "_").replace("-", "_") for col in df.columns]

    # --- Basic Info ---
    print("\n--- BASIC INFO ---")
    print(df.info())

    # --- Descriptive Statistics ---
    print("\n--- DESCRIPTION ---")
    print(df.describe())

    # --- Top Industries by Total Workers ---
    if "industry_name" in df.columns and "total_workers" in df.columns:
        print("\n--- TOP INDUSTRIES BY TOTAL WORKERS ---")
        top_industries = (
            df.groupby("industry_name")["total_workers"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        print(top_industries)
    else:
        print("\nColumns 'industry_name' or 'total_workers' not found. Skipping top industries.")

    # --- Top States by Total Workers ---
    if "state" in df.columns and "total_workers" in df.columns:
        print("\n--- TOP STATES BY TOTAL WORKERS ---")
        top_states = (
            df.groupby("state")["total_workers"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        print(top_states)
    else:
        print("\nColumns 'state' or 'total_workers' not found. Skipping top states.")

    # --- Optional: Top Industries by Main vs Marginal Workers ---
    if "main_total_persons" in df.columns and "marginal_total_persons" in df.columns and "industry_name" in df.columns:
        print("\n--- TOP INDUSTRIES BY MAIN vs MARGINAL WORKERS ---")
        summary = df.groupby("industry_name")[["main_total_persons", "marginal_total_persons"]].sum()
        summary["ratio_marginal_to_main"] = summary["marginal_total_persons"] / (summary["main_total_persons"] + 1e-6)
        top_ratio = summary.sort_values("ratio_marginal_to_main", ascending=False).head(10)
        print(top_ratio)
    else:
        print("\nColumns required for main vs marginal workers analysis not found. Skipping this section.")
