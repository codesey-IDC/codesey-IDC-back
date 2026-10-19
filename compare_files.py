import pandas as pd

def compare_parquets():
    orig_path = 'data/synthetic_idc_90d.parquet'
    new_path = 'output/final/idc_train_refined_2026.parquet'

    print(f"Comparing:\n1. {orig_path}\n2. {new_path}\n")

    try:
        df_orig = pd.read_parquet(orig_path)
        df_new = pd.read_parquet(new_path)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # 1. Row/Col counts
    print(f"Original shape: {df_orig.shape}")
    print(f"New shape:      {df_new.shape}")

    # 2. Column names
    orig_cols = set(df_orig.columns)
    new_cols = set(df_new.columns)

    if orig_cols == new_cols:
        print("\nColumns: Same names")
    else:
        print(f"\nColumns Added: {new_cols - orig_cols}")
        print(f"Columns Removed: {orig_cols - new_cols}")

    # 3. Data types
    dtypes_diff = False
    for col in (orig_cols & new_cols):
        if df_orig[col].dtype != df_new[col].dtype:
            print(f"Type mismatch in '{col}': {df_orig[col].dtype} -> {df_new[col].dtype}")
            dtypes_diff = True
    if not dtypes_diff:
        print("Data Types: Same types")

    # 4. Values (sample check)
    vals_diff = False
    for col in (orig_cols & new_cols):
        # Handle cases where values might be slightly different due to floating point or NaN
        if not df_orig[col].equals(df_new[col]):
            print(f"Value mismatch in '{col}'")
            vals_diff = True
            break
    if not vals_diff:
        print("Values: Identical content")

if __name__ == "__main__":
    compare_parquets()
