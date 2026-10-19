import pandas as pd
import json

def inspect_data():
    print("--- synthetic_idc_90d.parquet ---")
    try:
        df_synthetic = pd.read_parquet('data/synthetic_idc_90d.parquet')
        print(df_synthetic.head())
        print(df_synthetic.info())
    except Exception as e:
        print(f"Error reading synthetic_idc_90d.parquet: {e}")

    print("\n--- kepco_datacenter_2023.json ---")
    try:
        with open('data/external/kepco_datacenter_2023.json', 'r', encoding='utf-8') as f:
            kepco_data = json.load(f)
            print(json.dumps(kepco_data[:2] if isinstance(kepco_data, list) else list(kepco_data.items())[:2], indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error reading kepco_datacenter_2023.json: {e}")

    print("\n--- cleaned_datacenter_structured.json ---")
    try:
        with open('output/cleaned_datacenter_structured.json', 'r', encoding='utf-8') as f:
            structured_data = json.load(f)
            print(json.dumps(structured_data[:2] if isinstance(structured_data, list) else list(structured_data.items())[:2], indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error reading cleaned_datacenter_structured.json: {e}")

if __name__ == "__main__":
    inspect_data()
