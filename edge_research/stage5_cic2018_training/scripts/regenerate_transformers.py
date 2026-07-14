import os
import glob
import pandas as pd
import numpy as np
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
        # Since these files are large, we can read them efficiently
        df = pd.read_csv(f, low_memory=False)
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True)
    # Strip leading/trailing whitespaces from column names
    data.columns = data.columns.str.strip()
    return data

def preprocess():
    base_dir = "/Users/harshgoyal/Documents/BTP/BTP/edge_research/stage5_cic2018_training"
    input_pattern = "/Users/harshgoyal/Documents/BTP/BTP/dataset/cicids-2017/TrafficLabelling /Friday-WorkingHours-*.csv"
    
    df = load_data(input_pattern)
    print(f"Original shape: {df.shape}")

    # Remove specified columns
    cols_to_drop = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
    actual_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=actual_cols_to_drop, inplace=True)
    print(f"Dropped columns: {actual_cols_to_drop}")

    # Replace infinite values with NaN
    print("Replacing infinite values with NaN...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Handle missing values by dropping rows
    print("Handling missing values (dropping rows with NaN)...")
    df.dropna(inplace=True)
    print(f"Shape after cleaning: {df.shape}")

    # Separate features and label
    label_col = 'Label' if 'Label' in df.columns else [c for c in df.columns if 'label' in c.lower()][0]
    
    # We should map labels to target Classes: BENIGN, Bot, DDoS, PortScan
    # Friday files contain: BENIGN, DDoS, PortScan, and maybe Bot?
    # Let's check Friday labels:
    print("Unique labels in Friday files:", df[label_col].unique())
    
    # Let's map labels explicitly to stay consistent with the target label encoder classes
    # Target classes: ['BENIGN', 'Bot', 'DDoS', 'PortScan']
    # Friday files labels are usually 'BENIGN', 'DDoS', 'PortScan', and 'Bot' might not be there or might be there.
    # Let's make sure they map correctly:
    label_mapping = {
        'BENIGN': 'BENIGN',
        'DDoS': 'DDoS',
        'PortScan': 'PortScan',
        'Bot': 'Bot'
    }
    # Standardize spaces and casing
    df[label_col] = df[label_col].str.strip()
    
    X = df.drop(columns=[label_col])
    y = df[label_col]

    # Let's enforce that LabelEncoder has all 4 target classes: ['BENIGN', 'Bot', 'DDoS', 'PortScan']
    # Even if some class (like 'Bot') has 0 samples in Friday, we want the LabelEncoder to know about it.
    print("Fitting LabelEncoder on standard classes: ['BENIGN', 'Bot', 'DDoS', 'PortScan']")
    le = LabelEncoder()
    le.fit(['BENIGN', 'Bot', 'DDoS', 'PortScan'])
    y_encoded = le.transform(y)
    
    print(f"Class mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Split train and test sets
    print("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Normalize features
    print("Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save processed dataset and transformers
    out_dir = os.path.join(base_dir, "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    
    print("Saving processed datasets...")
    np.savez_compressed(os.path.join(out_dir, "cicids2017_processed.npz"), 
                        X_train=X_train_scaled.astype(np.float32), 
                        X_test=X_test_scaled.astype(np.float32), 
                        y_train=y_train.astype(np.int64), 
                        y_test=y_test.astype(np.int64))
    
    # Save the scaler and label encoder
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(le, os.path.join(out_dir, "label_encoder.pkl"))
    
    print(f"Preprocessing complete. Files saved to {out_dir}")

if __name__ == "__main__":
    preprocess()
