import pandas as pd
import numpy as np
import glob
import os

def explore_dataset(file_pattern):
    print(f"Loading files matching: {file_pattern}")
    files = glob.glob(file_pattern)
    if not files:
        print("No files found!")
        return

    dfs = []
    for f in files:
        print(f"Reading {os.path.basename(f)}...")
        df = pd.read_csv(f)
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True)
    
    # Strip leading/trailing whitespaces from column names
    data.columns = data.columns.str.strip()
    
    print("\n" + "="*50)
    print("DATASET EXPLORATION RESULTS")
    print("="*50)

    # 1. Total Features and Rows
    print(f"\n1. DATASET SHAPE")
    print(f"Total Rows: {data.shape[0]:,}")
    print(f"Total Features (Columns): {data.shape[1]}")
    
    # 2. Which features exist?
    print("\n2. FEATURES LIST")
    features = list(data.columns)
    print(", ".join(features))

    # 3. Missing values
    print("\n3. MISSING VALUES")
    missing = data.isnull().sum()
    missing_cols = missing[missing > 0]
    if missing_cols.empty:
        print("No missing values found.")
    else:
        for col, count in missing_cols.items():
            print(f"{col}: {count} missing values ({(count/len(data))*100:.2f}%)")

    # 4. Infinite values
    print("\n4. INFINITE VALUES")
    # Select only numeric columns for infinite check
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(data[numeric_cols]).sum()
    inf_cols = inf_counts[inf_counts > 0]
    if inf_cols.empty:
        print("No infinite values found.")
    else:
        for col, count in inf_cols.items():
            print(f"{col}: {count} infinite values ({(count/len(data))*100:.2f}%)")

    # 5. Attack Classes & Class Distribution
    print("\n5. CLASS DISTRIBUTION (ATTACKS)")
    if 'Label' in data.columns:
        label_col = 'Label'
    else:
        # Fallback in case column name is slightly different
        label_col = [c for c in data.columns if 'label' in c.lower()][0]
        
    distribution = data[label_col].value_counts()
    for attack, count in distribution.items():
        print(f"{attack}: {count:,} ({(count/len(data))*100:.2f}%)")

if __name__ == "__main__":
    # Path to the Friday datasets
    pattern = "../data/Friday-WorkingHours-*.csv"
    
    # Using absolute path just to be safe
    abs_pattern = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "Friday-WorkingHours-*.csv")
    explore_dataset(abs_pattern)
