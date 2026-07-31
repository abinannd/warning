import pandas as pd

file_path = r"C:\BRAIN-STORM\HT\warning\outbreak_detection_system\data\raw\indian_diseases_dataset_malabar-1.xlsx"
out_path = r"C:\BRAIN-STORM\HT\warning\outbreak_detection_system\data\interim\cleaned_data.pkl"

try:
    df = pd.read_excel(file_path)
    original_count = len(df)

    # Remove duplicates
    duplicates_count = df.duplicated().sum()
    df = df.drop_duplicates()

    # Trim whitespace from all string columns
    str_cols = df.select_dtypes(include=['object']).columns
    for col in str_cols:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Convert diagnosis_date
    df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'], errors='coerce')

    # Missing values check
    missing_subset = ['diagnosis_date', 'district', 'disease_name']
    missing_count = df[missing_subset].isna().any(axis=1).sum()
    df = df.dropna(subset=missing_subset)

    # Standardize capitalization
    df['district'] = df['district'].str.title()
    df['disease_name'] = df['disease_name'].str.title()

    final_count = len(df)

    print("--- CLEANING SUMMARY ---")
    print(f"Original Row Count: {original_count}")
    print(f"Duplicate Rows Removed: {duplicates_count}")
    print(f"Rows Removed Due to Missing Values: {missing_count}")
    print(f"Final Row Count: {final_count}")

    # Save to disk to simulate keeping it in memory
    df.to_pickle(out_path)
    print(f"\nCleaned DataFrame saved to interim storage for next step.")
except Exception as e:
    print(f"Error during data cleaning: {e}")
