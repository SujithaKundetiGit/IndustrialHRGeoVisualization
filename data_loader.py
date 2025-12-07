import os
import pandas as pd

def load_and_merge_csv(data_dir):
    """
    Load all CSV files from a folder, handle encoding issues, and merge into a single DataFrame.
    
    Args:
        data_dir (str): Path to the folder containing CSV files.
    
    Returns:
        pd.DataFrame or None: Merged DataFrame, or None if no files were loaded.
    """
    dfs = []
    if not os.path.exists(data_dir):
        print(f"Data directory does not exist: {data_dir}")
        return None

    csv_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return None

    for filename in csv_files:
        file_path = os.path.join(data_dir, filename)
        try:
            # Try reading with cp1252 encoding first (handles Windows Excel CSVs)
            df = pd.read_csv(file_path, encoding="cp1252")
            dfs.append(df)
            print(f"Loaded {filename} with {len(df)} rows.")
        except UnicodeDecodeError:
            print(f"UnicodeDecodeError in {filename}. Trying utf-8 with errors='replace'.")
            try:
                df = pd.read_csv(file_path, encoding="utf-8", errors="replace")
                dfs.append(df)
                print(f"Loaded {filename} with {len(df)} rows (utf-8 replace).")
            except Exception as e:
                print(f"Failed to read {filename}: {e}")
        except Exception as e:
            print(f"Failed to read {filename}: {e}")

    if not dfs:
        print("No CSV files could be loaded successfully.")
        return None

    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"Merged {len(dfs)} files with total {len(merged_df)} rows.")
    return merged_df
