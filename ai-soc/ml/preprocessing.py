import pandas as pd
import numpy as np
import glob
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_pattern):
    print(f"Loading files matching: {file_pattern}")
    files = glob.glob(file_pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching {file_pattern}")

    dfs = []
    for f in files:
        print(f"Reading {os.path.basename(f)}...")
        df = pd.read_csv(f)
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True)
    # Strip leading/trailing whitespaces from column names
    data.columns = data.columns.str.strip()
    return data

def preprocess():
    # 1. Load Data
    input_pattern = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "Friday-WorkingHours-*.csv")
    df = load_data(input_pattern)
    
    print(f"Original shape: {df.shape}")

    # 2. Remove specified columns
    cols_to_drop = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
    # Sometimes CICIDS2017 has slight variations in column names, let's be careful
    actual_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=actual_cols_to_drop, inplace=True)
    print(f"Dropped columns: {actual_cols_to_drop}")

    # 3. Critical Fix: Replace inf and -inf with NaN
    print("Replacing infinite values with NaN...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 4. Handle missing values
    # Given the missing/inf values are < 0.1%, dropping rows is the safest and cleanest approach
    print("Handling missing values (dropping rows with NaN)...")
    df.dropna(inplace=True)
    
    print(f"Shape after cleaning: {df.shape}")

    # 5. Separate features and label
    label_col = 'Label' if 'Label' in df.columns else [c for c in df.columns if 'label' in c.lower()][0]
    X = df.drop(columns=[label_col])
    y = df[label_col]

    # 6. Encode labels
    print("Encoding labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Print class mapping
    class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Class mapping: {class_mapping}")

    # 7. Split train and test sets
    print("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # 8. Normalize features
    print("Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 9. Save processed dataset and transformers
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    
    print("Saving processed datasets...")
    np.savez_compressed(os.path.join(out_dir, "cicids2017_processed.npz"), 
                        X_train=X_train_scaled, 
                        X_test=X_test_scaled, 
                        y_train=y_train, 
                        y_test=y_test)
    
    # Save the scaler and label encoder so they can be used during inference in production
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(le, os.path.join(out_dir, "label_encoder.pkl"))
    
    print(f"Preprocessing complete. Files saved to {out_dir}")

if __name__ == "__main__":
    preprocess()
