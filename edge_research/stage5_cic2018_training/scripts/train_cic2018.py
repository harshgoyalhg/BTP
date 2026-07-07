import os
# Fix for double OpenMP runtime linking aborts on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import glob
import time
import joblib
import psutil
import threading
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix, classification_report
)

import xgboost as xgb
import lightgbm as lgb

# 1. Setup Logging and Output Directories
BASE_DIR = "/Users/harshgoyal/Documents/BTP/BTP/edge_research/stage5_cic2018_training"
os.makedirs(os.path.join(BASE_DIR, "scripts"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "figures"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "tables"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "artifacts"), exist_ok=True)

log_format = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "logs", "training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("stage5_training")

logger.info("Initializing Stage 5 training execution on MacBook Air M4...")

# Memory tracking helper
class MemoryTracker:
    def __init__(self):
        self.max_memory = 0.0
        self.running = False
        self.thread = None

    def _track(self):
        process = psutil.Process(os.getpid())
        while self.running:
            try:
                mem = process.memory_info().rss / (1024 * 1024)  # MB
                if mem > self.max_memory:
                    self.max_memory = mem
            except Exception:
                pass
            time.sleep(0.05)

    def start(self):
        self.max_memory = 0.0
        self.running = True
        self.thread = threading.Thread(target=self._track, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        return self.max_memory

# 2. Configuration & Alignments
SCALER_FEATURES = [
    'Source Port', 'Destination Port', 'Protocol', 'Flow Duration', 'Total Fwd Packets',
    'Total Backward Packets', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total',
    'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
    'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s',
    'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std',
    'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
    'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Header Length.1',
    'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
    'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
    'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max',
    'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]

mapping_2018_to_2017 = {
    'Dst Port': 'Destination Port',
    'Protocol': 'Protocol',
    'Flow Duration': 'Flow Duration',
    'Tot Fwd Pkts': 'Total Fwd Packets',
    'Tot Bwd Pkts': 'Total Backward Packets',
    'TotLen Fwd Pkts': 'Total Length of Fwd Packets',
    'TotLen Bwd Pkts': 'Total Length of Bwd Packets',
    'Fwd Pkt Len Max': 'Fwd Packet Length Max',
    'Fwd Pkt Len Min': 'Fwd Packet Length Min',
    'Fwd Pkt Len Mean': 'Fwd Packet Length Mean',
    'Fwd Pkt Len Std': 'Fwd Packet Length Std',
    'Bwd Pkt Len Max': 'Bwd Packet Length Max',
    'Bwd Pkt Len Min': 'Bwd Packet Length Min',
    'Bwd Pkt Len Mean': 'Bwd Packet Length Mean',
    'Bwd Pkt Len Std': 'Bwd Packet Length Std',
    'Flow Byts/s': 'Flow Bytes/s',
    'Flow Pkts/s': 'Flow Packets/s',
    'Flow IAT Mean': 'Flow IAT Mean',
    'Flow IAT Std': 'Flow IAT Std',
    'Flow IAT Max': 'Flow IAT Max',
    'Flow IAT Min': 'Flow IAT Min',
    'Fwd IAT Tot': 'Fwd IAT Total',
    'Fwd IAT Mean': 'Fwd IAT Mean',
    'Fwd IAT Std': 'Fwd IAT Std',
    'Fwd IAT Max': 'Fwd IAT Max',
    'Fwd IAT Min': 'Fwd IAT Min',
    'Bwd IAT Tot': 'Bwd IAT Total',
    'Bwd IAT Mean': 'Bwd IAT Mean',
    'Bwd IAT Std': 'Bwd IAT Std',
    'Bwd IAT Max': 'Bwd IAT Max',
    'Bwd IAT Min': 'Bwd IAT Min',
    'Fwd PSH Flags': 'Fwd PSH Flags',
    'Bwd PSH Flags': 'Bwd PSH Flags',
    'Fwd URG Flags': 'Fwd URG Flags',
    'Bwd URG Flags': 'Bwd URG Flags',
    'Fwd Header Len': 'Fwd Header Length',
    'Bwd Header Len': 'Bwd Header Length',
    'Fwd Pkts/s': 'Fwd Packets/s',
    'Bwd Pkts/s': 'Bwd Packets/s',
    'Pkt Len Min': 'Min Packet Length',
    'Pkt Len Max': 'Max Packet Length',
    'Pkt Len Mean': 'Packet Length Mean',
    'Pkt Len Std': 'Packet Length Std',
    'Pkt Len Var': 'Packet Length Variance',
    'FIN Flag Cnt': 'FIN Flag Count',
    'SYN Flag Cnt': 'SYN Flag Count',
    'RST Flag Cnt': 'RST Flag Count',
    'PSH Flag Cnt': 'PSH Flag Count',
    'ACK Flag Cnt': 'ACK Flag Count',
    'URG Flag Cnt': 'URG Flag Count',
    'CWE Flag Count': 'CWE Flag Count',
    'ECE Flag Cnt': 'ECE Flag Count',
    'Down/Up Ratio': 'Down/Up Ratio',
    'Pkt Size Avg': 'Average Packet Size',
    'Fwd Seg Size Avg': 'Avg Fwd Segment Size',
    'Bwd Seg Size Avg': 'Avg Bwd Segment Size',
    'Fwd Byts/b Avg': 'Fwd Avg Bytes/Bulk',
    'Fwd Pkts/b Avg': 'Fwd Avg Packets/Bulk',
    'Fwd Blk Rate Avg': 'Fwd Avg Bulk Rate',
    'Bwd Byts/b Avg': 'Bwd Avg Bytes/Bulk',
    'Bwd Pkts/b Avg': 'Bwd Avg Packets/Bulk',
    'Bwd Blk Rate Avg': 'Bwd Avg Bulk Rate',
    'Subflow Fwd Pkts': 'Subflow Fwd Packets',
    'Subflow Fwd Byts': 'Subflow Fwd Bytes',
    'Subflow Bwd Pkts': 'Subflow Bwd Packets',
    'Subflow Bwd Byts': 'Subflow Bwd Bytes',
    'Init Fwd Win Byts': 'Init_Win_bytes_forward',
    'Init Bwd Win Byts': 'Init_Win_bytes_backward',
    'Fwd Act Data Pkts': 'act_data_pkt_fwd',
    'Fwd Seg Size Min': 'min_seg_size_forward',
    'Active Mean': 'Active Mean',
    'Active Std': 'Active Std',
    'Active Max': 'Active Max',
    'Active Min': 'Active Min',
    'Idle Mean': 'Idle Mean',
    'Idle Std': 'Idle Std',
    'Idle Max': 'Idle Max',
    'Idle Min': 'Idle Min',
    'Src Port': 'Source Port',
}

label_mapping = {
    'Benign': 'BENIGN',
    'Bot': 'Bot',
    'DoS attacks-GoldenEye': 'DDoS',
    'DoS attacks-Slowloris': 'DDoS',
    'DoS attacks-Hulk': 'DDoS',
    'DoS attacks-SlowHTTPTest': 'DDoS',
    'DDoS attacks-LOIC-HTTP': 'DDoS',
    'DDOS attack-HOIC': 'DDoS',
    'DDOS attack-LOIC-UDP': 'DDoS',
    'FTP-BruteForce': 'Bot',
    'SSH-Bruteforce': 'Bot',
    'Brute Force -Web': 'Bot',
    'Brute Force -XSS': 'Bot',
    'SQL Injection': 'Bot',
    'Infilteration': 'Bot'
}

# 3. Memory-Safe Chunked Loader
def load_and_preprocess_cic2018(dataset_dir):
    csv_files = sorted(glob.glob(os.path.join(dataset_dir, "*.csv")))
    logger.info(f"Scanning {len(csv_files)} files in dataset directory...")
    
    attack_dfs = []
    benign_dfs = []
    
    # Read files sequentially with memory-safe chunk sizes
    for f in csv_files:
        filename = os.path.basename(f)
        logger.info(f"Processing file: {filename}...")
        
        # Load in chunks of 200,000 rows
        chunk_idx = 0
        for chunk in pd.read_csv(f, chunksize=200000, low_memory=False):
            chunk_idx += 1
            # Clean columns
            chunk.columns = chunk.columns.str.strip()
            
            # Remove duplicated header rows
            if 'Label' in chunk.columns:
                chunk = chunk[chunk['Label'] != 'Label']
            else:
                continue
                
            # Rename columns
            renamed_cols = {c: mapping_2018_to_2017[c] for c in chunk.columns if c in mapping_2018_to_2017}
            chunk.rename(columns=renamed_cols, inplace=True)
            
            # Add Source Port if missing
            if 'Source Port' not in chunk.columns:
                chunk['Source Port'] = 0
                
            # Clone Fwd Header Length to Fwd Header Length.1
            if 'Fwd Header Length' in chunk.columns:
                chunk['Fwd Header Length.1'] = chunk['Fwd Header Length']
            else:
                chunk['Fwd Header Length'] = 0
                chunk['Fwd Header Length.1'] = 0
                
            # Keep only the scaler features and Label
            available_cols = [c for c in SCALER_FEATURES if c in chunk.columns]
            chunk = chunk[available_cols + ['Label']]
            
            # Add any entirely missing feature columns with 0
            for col in SCALER_FEATURES:
                if col not in chunk.columns:
                    chunk[col] = 0.0
            
            # Ensure correct column ordering
            chunk = chunk[SCALER_FEATURES + ['Label']]
            
            # Convert feature columns to float32, drop infs/NaNs
            features_df = chunk[SCALER_FEATURES].apply(pd.to_numeric, errors='coerce').astype(np.float32)
            features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            cleaned_chunk = pd.concat([features_df, chunk['Label']], axis=1)
            cleaned_chunk.dropna(inplace=True)
            
            if cleaned_chunk.empty:
                continue
                
            # Map labels
            cleaned_chunk['MappedLabel'] = cleaned_chunk['Label'].map(label_mapping)
            cleaned_chunk.dropna(subset=['MappedLabel'], inplace=True)
            cleaned_chunk.drop(columns=['Label'], inplace=True)
            cleaned_chunk.rename(columns={'MappedLabel': 'Label'}, inplace=True)
            
            # Separate Benign and Attacks
            benign_chunk = cleaned_chunk[cleaned_chunk['Label'] == 'BENIGN']
            attack_chunk = cleaned_chunk[cleaned_chunk['Label'] != 'BENIGN']
            
            if not attack_chunk.empty:
                attack_dfs.append(attack_chunk)
                
            if not benign_chunk.empty:
                # Downsample benign traffic within this chunk (keep ~2% to prevent RAM issues)
                sampled_benign = benign_chunk.sample(frac=0.02, random_state=42) if len(benign_chunk) > 1000 else benign_chunk
                benign_dfs.append(sampled_benign)
                
        logger.info(f"Finished sequential chunks for {filename}.")
        
    # Combine dataframes
    all_attacks = pd.concat(attack_dfs, ignore_index=True)
    all_benign = pd.concat(benign_dfs, ignore_index=True)
    
    logger.info(f"Loaded Attack rows: {len(all_attacks):,}")
    logger.info(f"Loaded Benign rows (downsampled): {len(all_benign):,}")
    
    combined_df = pd.concat([all_attacks, all_benign], ignore_index=True)
    logger.info(f"Combined clean dataset shape: {combined_df.shape}")
    
    return combined_df

# 4. Ingest and Preprocess Data or Load from Cache
npz_cache_path = os.path.join(BASE_DIR, "artifacts", "cicids2018_processed.npz")

if os.path.exists(npz_cache_path):
    logger.info("Step 1 & 2: Loading preprocessed scaled dataset from artifacts cache...")
    cached_data = np.load(npz_cache_path, allow_pickle=True)
    X_train_scaled = cached_data['X_train']
    X_val_scaled = cached_data['X_val']
    y_train = cached_data['y_train']
    y_val = cached_data['y_val']
    
    # Load label encoder to print details
    processed_dir = "/Users/harshgoyal/Documents/BTP/BTP/ai-soc/data/processed"
    le = joblib.load(os.path.join(processed_dir, "label_encoder.pkl"))
    
    logger.info(f"Cached Train shape: {X_train_scaled.shape}, val shape: {X_val_scaled.shape}")
else:
    logger.info("Step 1: Reading and pre-processing dataset...")
    cic2018_dir = "/Users/harshgoyal/Documents/BTP/BTP/dataset/cicids-2018/archive"
    raw_data = load_and_preprocess_cic2018(cic2018_dir)

    # Draw stratified subset of 500,000 rows (or all if smaller)
    target_size = min(500000, len(raw_data))
    logger.info(f"Step 2: Drawing representative stratified sample of {target_size:,} rows...")
    _, sampled_df = train_test_split(
        raw_data,
        test_size=target_size,
        random_state=42,
        stratify=raw_data['Label']
    )
    logger.info(f"Sampled class distribution:\n{sampled_df['Label'].value_counts()}")

    # Ingest 5 dummy rows for PortScan (which has 0 samples in CICIDS2018)
    logger.info("Injecting 5 dummy rows for missing 'PortScan' class...")
    dummy_rows = []
    for _ in range(5):
        dummy_row = {col: 0.0 for col in SCALER_FEATURES}
        dummy_row['Label'] = 'PortScan'
        dummy_rows.append(dummy_row)
    dummy_df = pd.DataFrame(dummy_rows)
    sampled_df = pd.concat([sampled_df, dummy_df], ignore_index=True)

    # Separate features and target
    X = sampled_df[SCALER_FEATURES]
    y = sampled_df['Label']

    # Train-Validation Split (80/20 stratified)
    X_train, X_val, y_train_labels, y_val_labels = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Load Stage 2 Scaler & Label Encoder
    logger.info("Loading preprocessing transformers from Stage 2...")
    processed_dir = "/Users/harshgoyal/Documents/BTP/BTP/ai-soc/data/processed"
    scaler = joblib.load(os.path.join(processed_dir, "scaler.pkl"))
    le = joblib.load(os.path.join(processed_dir, "label_encoder.pkl"))

    # Scale features
    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)

    # Encode targets
    y_train = le.transform(y_train_labels).astype(np.int64)
    y_val = le.transform(y_val_labels).astype(np.int64)

    # Save to artifacts cache
    logger.info("Saving preprocessed dataset to cache for fast future runs...")
    np.savez_compressed(npz_cache_path, X_train=X_train_scaled, X_val=X_val_scaled, y_train=y_train, y_val=y_val)

# Print final confirmation
logger.info(f"Train set scaled shape: {X_train_scaled.shape}, encoded labels: {np.bincount(y_train)}")
logger.info(f"Val set scaled shape: {X_val_scaled.shape}, encoded labels: {np.bincount(y_val)}")
logger.info(f"Label encoder target classes: {le.classes_}")

# 4.5. Realistic PortScan Injection (from 2017 Dataset)
# Discard dummy PortScan samples (label 3) if any exist in the loaded/cached dataset
train_non_dummy_mask = (y_train != 3)
X_train_scaled = X_train_scaled[train_non_dummy_mask]
y_train = y_train[train_non_dummy_mask]

val_non_dummy_mask = (y_val != 3)
X_val_scaled = X_val_scaled[val_non_dummy_mask]
y_val = y_val[val_non_dummy_mask]

logger.info(f"Removed dummy PortScan rows. Train shape: {X_train_scaled.shape}, Val shape: {X_val_scaled.shape}")

# Load realistic PortScan samples from 2017 dataset
logger.info("Injecting 10,000 real PortScan samples from CICIDS2017 dataset...")
data_2017_path = "/Users/harshgoyal/Documents/BTP/BTP/ai-soc/data/processed/cicids2017_processed.npz"
data_2017 = np.load(data_2017_path)
X_train_2017 = data_2017['X_train']
y_train_2017 = data_2017['y_train']

# Extract real PortScan samples (label index is 3)
portscan_indices = np.where(y_train_2017 == 3)[0]
np.random.seed(42)
selected_indices = np.random.choice(portscan_indices, size=10000, replace=False)
X_portscan = X_train_2017[selected_indices]
y_portscan = y_train_2017[selected_indices]

# Split 80/20 into train/val
X_ps_train, X_ps_val, y_ps_train, y_ps_val = train_test_split(
    X_portscan, y_portscan, test_size=0.2, random_state=42, stratify=y_portscan
)

# Concatenate with the main datasets
X_train_scaled = np.vstack([X_train_scaled, X_ps_train])
y_train = np.concatenate([y_train, y_ps_train])
X_val_scaled = np.vstack([X_val_scaled, X_ps_val])
y_val = np.concatenate([y_val, y_ps_val])

logger.info(f"After PortScan injection - Train shape: {X_train_scaled.shape}, bincount: {np.bincount(y_train)}")
logger.info(f"After PortScan injection - Val shape: {X_val_scaled.shape}, bincount: {np.bincount(y_val)}")

# 5. Define PyTorch MLP Architecture
class MLPClassifierNet(nn.Module):
    def __init__(self, input_dim=80, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# Model list container
results_metrics = []
trained_models = {}

# 6. Evaluation Function
def evaluate_model(name, model_obj, X_test, y_test, is_pytorch=False, device='cpu'):
    logger.info(f"Evaluating {name}...")
    start_inf = time.time()
    
    if is_pytorch:
        model_obj.eval()
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = model_obj(X_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            y_pred = np.argmax(probs, axis=1)
    else:
        y_pred = model_obj.predict(X_test)
        probs = model_obj.predict_proba(X_test)
        
    end_inf = time.time()
    inference_time = end_inf - start_inf
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # Multiclass ROC & PR AUC (One-Vs-Rest)
    try:
        roc_auc = roc_auc_score(y_test, probs, multi_class='ovr', average='macro')
    except Exception as e:
        logger.warning(f"Could not compute ROC-AUC for {name}: {e}")
        roc_auc = np.nan
        
    try:
        # Compute PR AUC for each class and average
        pr_aucs = []
        for i in range(len(le.classes_)):
            y_true_bin = (y_test == i).astype(int)
            y_prob_bin = probs[:, i]
            precision_vals, recall_vals, _ = precision_recall_curve(y_true_bin, y_prob_bin)
            pr_aucs.append(auc(recall_vals, precision_vals))
        pr_auc = np.mean(pr_aucs)
    except Exception as e:
        logger.warning(f"Could not compute PR-AUC for {name}: {e}")
        pr_auc = np.nan
        
    logger.info(f"{name} Evaluation: Acc={acc:.4f}, Weighted-F1={weighted_f1:.4f}, InfTime={inference_time:.3f}s")
    
    # Generate confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "figures", f"confusion_matrix_{name.lower().replace(' ', '_')}.png"), dpi=300)
    plt.close()
    
    # Save predictions
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Balanced Accuracy": bal_acc,
        "MCC": mcc,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Inference Time": inference_time,
        "CM": cm,
        "probs": probs,
        "preds": y_pred
    }

# 7. Training Execution
# A. LOGISTIC REGRESSION
logger.info("\n=== Training Logistic Regression ===")
lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0, class_weight='balanced')
mem_tracker = MemoryTracker()
mem_tracker.start()
start_time = time.time()
lr.fit(X_train_scaled, y_train)
end_time = time.time()
peak_ram = mem_tracker.stop()
train_time = end_time - start_time
logger.info(f"LR completed in {train_time:.2f}s. Peak RAM: {peak_ram:.2f} MB")

# Save model
model_path = os.path.join(BASE_DIR, "models", "logistic_regression.pkl")
joblib.dump(lr, model_path)
model_size = os.path.getsize(model_path) / (1024 * 1024) # MB

# Evaluate
evals = evaluate_model("Logistic Regression", lr, X_val_scaled, y_val)
results_metrics.append({
    "Model": "Logistic Regression",
    "Train Time (s)": train_time,
    "Inference Time (s)": evals["Inference Time"],
    "Peak RAM (MB)": peak_ram,
    "Model Size (MB)": model_size,
    **{k: v for k, v in evals.items() if k not in ["CM", "probs", "preds", "Inference Time"]}
})
trained_models["Logistic Regression"] = (lr, evals)


# B. RANDOM FOREST
logger.info("\n=== Training Random Forest ===")
rf = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42, class_weight='balanced')
mem_tracker = MemoryTracker()
mem_tracker.start()
start_time = time.time()
rf.fit(X_train_scaled, y_train)
end_time = time.time()
peak_ram = mem_tracker.stop()
train_time = end_time - start_time
logger.info(f"RF completed in {train_time:.2f}s. Peak RAM: {peak_ram:.2f} MB")

# Save model
model_path = os.path.join(BASE_DIR, "models", "random_forest.pkl")
joblib.dump(rf, model_path)
model_size = os.path.getsize(model_path) / (1024 * 1024)

# Evaluate
evals = evaluate_model("Random Forest", rf, X_val_scaled, y_val)
results_metrics.append({
    "Model": "Random Forest",
    "Train Time (s)": train_time,
    "Inference Time (s)": evals["Inference Time"],
    "Peak RAM (MB)": peak_ram,
    "Model Size (MB)": model_size,
    **{k: v for k, v in evals.items() if k not in ["CM", "probs", "preds", "Inference Time"]}
})
trained_models["Random Forest"] = (rf, evals)


# C. EXTRA TREES
logger.info("\n=== Training Extra Trees ===")
et = ExtraTreesClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42, class_weight='balanced')
mem_tracker = MemoryTracker()
mem_tracker.start()
start_time = time.time()
et.fit(X_train_scaled, y_train)
end_time = time.time()
peak_ram = mem_tracker.stop()
train_time = end_time - start_time
logger.info(f"ET completed in {train_time:.2f}s. Peak RAM: {peak_ram:.2f} MB")

# Save model
model_path = os.path.join(BASE_DIR, "models", "extra_trees.pkl")
joblib.dump(et, model_path)
model_size = os.path.getsize(model_path) / (1024 * 1024)

# Evaluate
evals = evaluate_model("Extra Trees", et, X_val_scaled, y_val)
results_metrics.append({
    "Model": "Extra Trees",
    "Train Time (s)": train_time,
    "Inference Time (s)": evals["Inference Time"],
    "Peak RAM (MB)": peak_ram,
    "Model Size (MB)": model_size,
    **{k: v for k, v in evals.items() if k not in ["CM", "probs", "preds", "Inference Time"]}
})
trained_models["Extra Trees"] = (et, evals)


# D. XGBOOST
logger.info("\n=== Training XGBoost (optimized CPU) ===")
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    device="cpu",
    n_jobs=-1,
    random_state=42
)
mem_tracker = MemoryTracker()
mem_tracker.start()
start_time = time.time()
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
xgb_clf.fit(X_train_scaled, y_train, sample_weight=sample_weights)
end_time = time.time()
peak_ram = mem_tracker.stop()
train_time = end_time - start_time
logger.info(f"XGB completed in {train_time:.2f}s. Peak RAM: {peak_ram:.2f} MB")

# Save model
model_path = os.path.join(BASE_DIR, "models", "xgboost.pkl")
joblib.dump(xgb_clf, model_path)
model_size = os.path.getsize(model_path) / (1024 * 1024)

# Evaluate
evals = evaluate_model("XGBoost", xgb_clf, X_val_scaled, y_val)
results_metrics.append({
    "Model": "XGBoost",
    "Train Time (s)": train_time,
    "Inference Time (s)": evals["Inference Time"],
    "Peak RAM (MB)": peak_ram,
    "Model Size (MB)": model_size,
    **{k: v for k, v in evals.items() if k not in ["CM", "probs", "preds", "Inference Time"]}
})
trained_models["XGBoost"] = (xgb_clf, evals)


# E. LIGHTGBM
logger.info("\n=== Training LightGBM (optimized CPU) ===")
lgb_clf = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    n_jobs=1,
    random_state=42,
    verbose=-1,
    class_weight='balanced'
)
mem_tracker = MemoryTracker()
mem_tracker.start()
start_time = time.time()
lgb_clf.fit(X_train_scaled, y_train)
end_time = time.time()
peak_ram = mem_tracker.stop()
train_time = end_time - start_time
logger.info(f"LGBM completed in {train_time:.2f}s. Peak RAM: {peak_ram:.2f} MB")

# Save model
model_path = os.path.join(BASE_DIR, "models", "lightgbm.pkl")
joblib.dump(lgb_clf, model_path)
model_size = os.path.getsize(model_path) / (1024 * 1024)

# Evaluate
evals = evaluate_model("LightGBM", lgb_clf, X_val_scaled, y_val)
results_metrics.append({
    "Model": "LightGBM",
    "Train Time (s)": train_time,
    "Inference Time (s)": evals["Inference Time"],
    "Peak RAM (MB)": peak_ram,
    "Model Size (MB)": model_size,
    **{k: v for k, v in evals.items() if k not in ["CM", "probs", "preds", "Inference Time"]}
})
trained_models["LightGBM"] = (lgb_clf, evals)


# F. MULTI-LAYER PERCEPTRON (PyTorch on Apple Silicon MPS)
logger.info("\n=== Training MLP (PyTorch on MPS) ===")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Apple Silicon MPS Acceleration Status: {'ENABLED' if device.type == 'mps' else 'DISABLED (Falling back to CPU)'}")

mlp = MLPClassifierNet(input_dim=80, num_classes=4).to(device)

# Compute class weights for cost-sensitive learning
from sklearn.utils.class_weight import compute_class_weight
class_weights_val = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights_val, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(mlp.parameters(), lr=0.002)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# Convert to Tensor Datasets (float32 throughout, pin_memory=False, num_workers=0)
train_dataset = TensorDataset(
    torch.tensor(X_train_scaled, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.int64)
)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=0, pin_memory=False)

mem_tracker = MemoryTracker()
mem_tracker.start()
start_time = time.time()

# Train loops (30 epochs)
mlp.train()
for epoch in range(30):
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = mlp(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_x.size(0)
    epoch_loss /= len(train_dataset)
    scheduler.step()
    if (epoch + 1) % 5 == 0 or epoch == 0:
        logger.info(f"  Epoch {epoch+1}/30 - Loss: {epoch_loss:.4f}")

end_time = time.time()
peak_ram = mem_tracker.stop()
train_time = end_time - start_time
logger.info(f"MLP training completed in {train_time:.2f}s. Peak RAM: {peak_ram:.2f} MB")

# Save model parameters
model_path = os.path.join(BASE_DIR, "models", "mlp.pt")
torch.save(mlp.state_dict(), model_path)
model_size = os.path.getsize(model_path) / (1024 * 1024)

# Evaluate MLP
evals = evaluate_model("MLP (PyTorch)", mlp, X_val_scaled, y_val, is_pytorch=True, device=device)
results_metrics.append({
    "Model": "MLP (PyTorch)",
    "Train Time (s)": train_time,
    "Inference Time (s)": evals["Inference Time"],
    "Peak RAM (MB)": peak_ram,
    "Model Size (MB)": model_size,
    **{k: v for k, v in evals.items() if k not in ["CM", "probs", "preds", "Inference Time"]}
})
trained_models["MLP (PyTorch)"] = (mlp, evals)


# 8. Generate Tables & Reports
logger.info("\n=== Compiling Results Tables ===")
df_metrics = pd.DataFrame(results_metrics)
df_metrics.to_csv(os.path.join(BASE_DIR, "tables", "training_results.csv"), index=False)
df_metrics.to_csv(os.path.join(BASE_DIR, "tables", "model_comparison.csv"), index=False)

# Resource usage extraction
df_resources = df_metrics[["Model", "Train Time (s)", "Inference Time (s)", "Peak RAM (MB)", "Model Size (MB)"]]
df_resources.to_csv(os.path.join(BASE_DIR, "tables", "resource_usage.csv"), index=False)

# Classification reports
reports = []
for model_name, (_, evals) in trained_models.items():
    pred_labels = le.classes_[evals["preds"]]
    true_labels = le.classes_[y_val]
    report_dict = classification_report(true_labels, pred_labels, output_dict=True)
    for cls_name, cls_metrics in report_dict.items():
        if isinstance(cls_metrics, dict):
            reports.append({
                "Model": model_name,
                "Class": cls_name,
                "Precision": cls_metrics["precision"],
                "Recall": cls_metrics["recall"],
                "F1-Score": cls_metrics["f1-score"],
                "Support": cls_metrics["support"]
            })
pd.DataFrame(reports).to_csv(os.path.join(BASE_DIR, "tables", "classification_reports.csv"), index=False)

# Feature importance computation
feature_importances = {"Feature": SCALER_FEATURES}
for model_name, (model_obj, _) in trained_models.items():
    if hasattr(model_obj, "feature_importances_"):
        feature_importances[model_name] = model_obj.feature_importances_
df_fi = pd.DataFrame(feature_importances)
df_fi.to_csv(os.path.join(BASE_DIR, "tables", "feature_importance.csv"), index=False)


# 9. Plotting Visualizations
logger.info("Generating publication-quality visualization figures...")

# A. Multiclass ROC Curves
plt.figure(figsize=(8, 6))
for model_name, (_, evals) in trained_models.items():
    probs = evals["probs"]
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    for i in range(len(le.classes_)):
        y_true_bin = (y_val == i).astype(int)
        y_prob_bin = probs[:, i]
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true_bin, y_prob_bin)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[0] = 0.0
    macro_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, label=f"{model_name} (Macro AUC = {macro_auc:.3f})", lw=2)

plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Macro-Average ROC Curves (CICIDS2018 Validation)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "figures", "roc_curve.png"), dpi=300)
plt.close()


# B. Multiclass PR Curves
plt.figure(figsize=(8, 6))
for model_name, (_, evals) in trained_models.items():
    probs = evals["probs"]
    mean_recall = np.linspace(0, 1, 100)
    precisions = []
    for i in range(len(le.classes_)):
        y_true_bin = (y_val == i).astype(int)
        y_prob_bin = probs[:, i]
        p_vals, r_vals, _ = precision_recall_curve(y_true_bin, y_prob_bin)
        precisions.append(np.interp(mean_recall, r_vals[::-1], p_vals[::-1]))
    mean_precision = np.mean(precisions, axis=0)
    macro_pr_auc = auc(mean_recall, mean_precision)
    plt.plot(mean_recall, mean_precision, label=f"{model_name} (PR AUC = {macro_pr_auc:.3f})", lw=2)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Macro-Average Precision-Recall Curves (CICIDS2018)')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "figures", "pr_curve.png"), dpi=300)
plt.close()


# C. Feature Importance
top_n = 15
if "XGBoost" in df_fi.columns:
    df_fi_sorted = df_fi.sort_values(by="XGBoost", ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_fi_sorted, y="Feature", x="XGBoost", palette="viridis")
    plt.title(f"Top {top_n} Feature Importances (XGBoost)")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "figures", "feature_importance.png"), dpi=300)
    plt.close()


# D. Resource Comparisons
metrics_to_plot = ["Train Time (s)", "Inference Time (s)", "Peak RAM (MB)", "Model Size (MB)"]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, col in enumerate(metrics_to_plot):
    sns.barplot(data=df_metrics, x="Model", y=col, ax=axes[idx], palette="muted")
    axes[idx].set_title(col)
    axes[idx].set_xlabel("")
    axes[idx].grid(True, alpha=0.3)
    for label in axes[idx].get_xticklabels():
        label.set_rotation(15)

plt.suptitle("Resource Profiling Comparison - Apple Silicon M4", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "figures", "metrics_comparison.png"), dpi=300)
plt.close()

logger.info("Successfully completed Stage 5 pipeline training and analysis execution!")
