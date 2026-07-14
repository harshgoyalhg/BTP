import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import joblib
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix, classification_report
)

# 1. Setup Logging and Directories
BASE_DIR = "/Users/harshgoyal/Documents/BTP/BTP/edge_research/stage5_cic2018_training"
os.makedirs(os.path.join(BASE_DIR, "figures", "lycos"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "tables"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "artifacts"), exist_ok=True)

log_format = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "logs", "lycos_validation.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("lycos_eval")

logger.info("Initializing Cross-Dataset Validation on Lycos 2018 Dataset...")

# Target Classes in Label Encoder
CLASSES = ['BENIGN', 'Bot', 'DDoS', 'PortScan']

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

lycos_mapping = {
    'dst_port': 'Destination Port',
    'ip_prot': 'Protocol',
    'flow_duration': 'Flow Duration',
    'fwd_pkt_cnt': 'Total Fwd Packets',
    'bwd_pkt_cnt': 'Total Backward Packets',
    'fwd_pkt_len_tot': 'Total Length of Fwd Packets',
    'bwd_pkt_len_tot': 'Total Length of Bwd Packets',
    'fwd_pkt_len_max': 'Fwd Packet Length Max',
    'fwd_pkt_len_min': 'Fwd Packet Length Min',
    'fwd_pkt_len_mean': 'Fwd Packet Length Mean',
    'fwd_pkt_len_std': 'Fwd Packet Length Std',
    'bwd_pkt_len_max': 'Bwd Packet Length Max',
    'bwd_pkt_len_min': 'Bwd Packet Length Min',
    'bwd_pkt_len_mean': 'Bwd Packet Length Mean',
    'bwd_pkt_len_std': 'Bwd Packet Length Std',
    'bytes_per_s': 'Flow Bytes/s',
    'pkt_per_s': 'Flow Packets/s',
    'iat_mean': 'Flow IAT Mean',
    'iat_std': 'Flow IAT Std',
    'iat_max': 'Flow IAT Max',
    'iat_min': 'Flow IAT Min',
    'fwd_iat_tot': 'Fwd IAT Total',
    'fwd_iat_mean': 'Fwd IAT Mean',
    'fwd_iat_std': 'Fwd IAT Std',
    'fwd_iat_max': 'Fwd IAT Max',
    'fwd_iat_min': 'Fwd IAT Min',
    'bwd_iat_tot': 'Bwd IAT Total',
    'bwd_iat_mean': 'Bwd IAT Mean',
    'bwd_iat_std': 'Bwd IAT Std',
    'bwd_iat_max': 'Bwd IAT Max',
    'bwd_iat_min': 'Bwd IAT Min',
    'fwd_flag_psh': 'Fwd PSH Flags',
    'bwd_flag_psh': 'Bwd PSH Flags',
    'fwd_flag_urg': 'Fwd URG Flags',
    'bwd_flag_urg': 'Bwd URG Flags',
    'fwd_pkt_hdr_len_tot': 'Fwd Header Length',
    'bwd_pkt_hdr_len_tot': 'Bwd Header Length',
    'fwd_pkt_per_s': 'Fwd Packets/s',
    'bwd_pkt_per_s': 'Bwd Packets/s',
    'pkt_len_min': 'Min Packet Length',
    'pkt_len_max': 'Max Packet Length',
    'pkt_len_mean': 'Packet Length Mean',
    'pkt_len_var': 'Packet Length Variance',
    'pkt_len_std': 'Packet Length Std',
    'flag_fin': 'FIN Flag Count',
    'flag_SYN': 'SYN Flag Count',
    'flag_rst': 'RST Flag Count',
    'flag_psh': 'PSH Flag Count',
    'flag_ack': 'ACK Flag Count',
    'flag_urg': 'URG Flag Count',
    'flag_cwr': 'CWE Flag Count',
    'flag_ece': 'ECE Flag Count',
    'down_up_ratio': 'Down/Up Ratio',
    'fwd_bulk_bytes_mean': 'Fwd Avg Bytes/Bulk',
    'fwd_bulk_pkt_mean': 'Fwd Avg Packets/Bulk',
    'fwd_bulk_rate_mean': 'Fwd Avg Bulk Rate',
    'bwd_bulk_bytes_mean': 'Bwd Avg Bytes/Bulk',
    'bwd_bulk_pkt_mean': 'Bwd Avg Packets/Bulk',
    'bwd_bulk_rate_mean': 'Bwd Avg Bulk Rate',
    'fwd_subflow_pkt_mean': 'Subflow Fwd Packets',
    'fwd_subflow_bytes_mean': 'Subflow Fwd Bytes',
    'bwd_subflow_pkt_mean': 'Subflow Bwd Packets',
    'bwd_subflow_bytes_mean': 'Subflow Bwd Bytes',
    'fwd_tcp_init_win_bytes': 'Init_Win_bytes_forward',
    'bwd_tcp_init_win_bytes': 'Init_Win_bytes_backward',
    'fwd_non_empty_pkt_cnt': 'act_data_pkt_fwd',
    'fwd_pkt_hdr_len_min': 'min_seg_size_forward',
    'active_mean': 'Active Mean',
    'active_std': 'Active Std',
    'active_max': 'Active Max',
    'active_min': 'Active Min',
    'idle_mean': 'Idle Mean',
    'idle_std': 'Idle Std',
    'idle_max': 'Idle Max',
    'idle_min': 'Idle Min'
}

lycos_label_mapping = {
    'Benign': 'BENIGN',
    'Bot': 'Bot',
    'FTP-Patator': 'Bot',
    'SSH-Patator': 'Bot',
    'Web Attack - Brute Force': 'Bot',
    'Web Attack - Sql Injection': 'Bot',
    'Web Attack - XSS': 'Bot',
    'DDoS HOIC': 'DDoS',
    'DDoS LOIC-HTTP': 'DDoS',
    'DDoS LOIC-UDP': 'DDoS',
    'DoS GoldenEye': 'DDoS',
    'DoS Hulk': 'DDoS',
    'DoS Slowhttptest': 'DDoS',
    'DoS Slowloris': 'DDoS'
}

# 2. Sequential Load & Downsample or Cache Ingestion
lycos_cache_path = os.path.join(BASE_DIR, "artifacts", "lycos_processed.npz")

if os.path.exists(lycos_cache_path):
    logger.info("Loading preprocessed Lycos dataset from cache...")
    cached = np.load(lycos_cache_path)
    X_test = cached['X_test']
    y_test = cached['y_test']
    logger.info(f"Loaded from cache. Shape: {X_test.shape}, Labels: {np.bincount(y_test)}")
else:
    logger.info("Ingesting Lycos CSV chunk-by-chunk and downsampling...")
    csv_path = "/Users/harshgoyal/Documents/BTP/BTP/dataset/lycos-2018/LycoS-Unicas-IDS2018.csv"
    chunk_list = []
    chunk_idx = 0
    for chunk in pd.read_csv(csv_path, chunksize=500000, low_memory=False):
        chunk_idx += 1
        
        # Clean headers
        chunk.columns = chunk.columns.str.strip()
        
        # Map labels
        chunk['MappedLabel'] = chunk['label'].map(lycos_label_mapping)
        chunk.dropna(subset=['MappedLabel'], inplace=True)
        chunk.drop(columns=['label'], inplace=True)
        chunk.rename(columns={'MappedLabel': 'Label'}, inplace=True)
        
        # Downsample Benign (keep 1%)
        benign_df = chunk[chunk['Label'] == 'BENIGN']
        attack_df = chunk[chunk['Label'] != 'BENIGN']
        
        if len(benign_df) > 0:
            benign_df = benign_df.sample(frac=0.01, random_state=42)
        # Downsample Attacks (keep 2.5% to balance)
        if len(attack_df) > 0:
            attack_df = attack_df.sample(frac=0.025, random_state=42)
            
        sampled = pd.concat([benign_df, attack_df], ignore_index=True)
        chunk_list.append(sampled)
        logger.info(f"Processed chunk {chunk_idx}.")
        
    combined_df = pd.concat(chunk_list, ignore_index=True)
    logger.info(f"Combined sample shape: {combined_df.shape}")
    
    # Feature Alignment
    X_raw = combined_df[list(lycos_mapping.keys())].copy()
    y_raw = combined_df['Label']
    
    # Rename columns to 2017 schema
    X_raw.rename(columns=lycos_mapping, inplace=True)
    
    # Fill missing Source Port
    X_raw['Source Port'] = 0.0
    
    # Populate duplicate columns & computed metrics
    X_raw['Average Packet Size'] = X_raw['Packet Length Mean']
    X_raw['Avg Fwd Segment Size'] = X_raw['Fwd Packet Length Mean']
    X_raw['Avg Bwd Segment Size'] = X_raw['Bwd Packet Length Mean']
    X_raw['Fwd Header Length.1'] = X_raw['Fwd Header Length']
    
    # Ensure correct ordering
    X_raw = X_raw[SCALER_FEATURES]
    
    # Cast to float32 and resolve inf/nan
    X_raw = X_raw.apply(pd.to_numeric, errors='coerce').astype(np.float32)
    X_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    clean_idx = X_raw.notna().all(axis=1)
    X_clean = X_raw[clean_idx].copy()
    y_clean = y_raw[clean_idx].copy()
    
    # Scale Features
    processed_dir = os.path.join(BASE_DIR, "artifacts")
    scaler = joblib.load(os.path.join(processed_dir, "scaler.pkl"))
    le = joblib.load(os.path.join(processed_dir, "label_encoder.pkl"))
    
    X_test = scaler.transform(X_clean).astype(np.float32)
    y_test = le.transform(y_clean).astype(np.int64)
    
    logger.info(f"Saving preprocessed Lycos arrays to cache...")
    np.savez_compressed(lycos_cache_path, X_test=X_test, y_test=y_test)
    logger.info(f"Final Lycos test set shape: {X_test.shape}, labels: {np.bincount(y_test)}")
# 2.5. Feature Engineering (Unscaling, computing ratios, scaling, and appending)
logger.info("Applying feature engineering (ratios on raw features) for Lycos test set...")
# Load scalers
processed_dir = os.path.join(BASE_DIR, "artifacts")
scaler = joblib.load(os.path.join(processed_dir, "scaler.pkl"))
engineered_scaler = joblib.load(os.path.join(processed_dir, "engineered_scaler.pkl"))

# Reconstruct raw values
X_raw = scaler.inverse_transform(X_test)
idx_fwd_pkts = SCALER_FEATURES.index('Total Fwd Packets')
idx_bwd_pkts = SCALER_FEATURES.index('Total Backward Packets')
idx_fwd_len = SCALER_FEATURES.index('Total Length of Fwd Packets')
idx_bwd_len = SCALER_FEATURES.index('Total Length of Bwd Packets')

# Compute ratios
fwd_pkts = X_raw[:, idx_fwd_pkts]
bwd_pkts = X_raw[:, idx_bwd_pkts]
fwd_len = X_raw[:, idx_fwd_len]
bwd_len = X_raw[:, idx_bwd_len]

pkt_ratio = fwd_pkts / (bwd_pkts + 1.0)
len_ratio = fwd_len / (bwd_len + 1.0)

new_features = np.column_stack([pkt_ratio, len_ratio])
new_features_scaled = engineered_scaler.transform(new_features)

X_test = np.column_stack([X_test, new_features_scaled])
logger.info(f"After Feature Engineering - Lycos test set shape: {X_test.shape}")

# 3. Define Neural Net Architecture
class MLPClassifierNet(nn.Module):
    def __init__(self, input_dim=82, num_classes=4):
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

# 4. Evaluation Loop
results = []
trained_models = {}

def evaluate_model(name, model_obj, X_eval, y_eval, is_pytorch=False, device='cpu'):
    logger.info(f"Evaluating {name} on Lycos...")
    start_time = time.time()
    
    if is_pytorch:
        model_obj.eval()
        X_tensor = torch.tensor(X_eval, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = model_obj(X_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            y_pred = np.argmax(probs, axis=1)
    else:
        y_pred = model_obj.predict(X_eval)
        if hasattr(model_obj, "predict_proba"):
            probs = model_obj.predict_proba(X_eval)
        else:
            # Fallback for models like LinearSVC without predict_proba
            dec = model_obj.decision_function(X_eval)
            if len(dec.shape) == 1:
                probs_bin = 1 / (1 + np.exp(-dec))
                probs = np.column_stack([1 - probs_bin, probs_bin])
            else:
                exp_dec = np.exp(dec - np.max(dec, axis=1, keepdims=True))
                probs = exp_dec / np.sum(exp_dec, axis=1, keepdims=True)
        
    end_time = time.time()
    inference_time = end_time - start_time
    
    acc = accuracy_score(y_eval, y_pred)
    prec = precision_score(y_eval, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_eval, y_pred, average='weighted', zero_division=0)
    macro_f1 = f1_score(y_eval, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_eval, y_pred, average='weighted', zero_division=0)
    bal_acc = balanced_accuracy_score(y_eval, y_pred)
    mcc = matthews_corrcoef(y_eval, y_pred)
    
    # Multiclass ROC & PR AUC (One-Vs-Rest)
    try:
        roc_auc = roc_auc_score(y_eval, probs, multi_class='ovr', average='macro')
    except Exception as e:
        roc_auc = np.nan
        
    try:
        pr_aucs = []
        for i in range(len(CLASSES)):
            y_true_bin = (y_eval == i).astype(int)
            y_prob_bin = probs[:, i]
            precision_vals, recall_vals, _ = precision_recall_curve(y_true_bin, y_prob_bin)
            pr_aucs.append(auc(recall_vals, precision_vals))
        pr_auc = np.mean(pr_aucs)
    except Exception as e:
        pr_auc = np.nan
        
    logger.info(f"{name} Results: Acc={acc:.4f}, Weighted-F1={weighted_f1:.4f}, Macro-F1={macro_f1:.4f}, InfTime={inference_time:.3f}s")
    
    # Save Confusion Matrix Heatmap
    cm = confusion_matrix(y_eval, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"Lycos Cross-Dataset Confusion Matrix - {name}")
    plt.ylabel('Actual (Lycos)')
    plt.xlabel('Predicted (by CICIDS2018 model)')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "figures", "lycos", f"confusion_matrix_{name.lower().replace(' ', '_')}.png"), dpi=300)
    plt.close()
    
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
        "Inference Time (s)": inference_time,
        "probs": probs,
        "preds": y_pred
    }

# 5. Load and Evaluate Models Sequentially
models_list = [
    ("Logistic Regression", "logistic_regression.pkl", False),
    ("Random Forest", "random_forest.pkl", False),
    ("Extra Trees", "extra_trees.pkl", False),
    ("XGBoost", "xgboost.pkl", False),
    ("LightGBM", "lightgbm.pkl", False),
    ("MLP (PyTorch)", "mlp.pt", True),
    ("SVM (Linear)", "svm_linear.pkl", False),
    ("SVM (RBF Kernel)", "svm_rbf.pkl", False),
]

for name, filename, is_pytorch in models_list:
    model_path = os.path.join(BASE_DIR, "models", filename)
    if not os.path.exists(model_path):
        continue
        
    if is_pytorch:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = MLPClassifierNet(input_dim=X_test.shape[1], num_classes=4).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        evals = evaluate_model(name, model, X_test, y_test, is_pytorch=True, device=device)
        trained_models[name] = (model, evals)
    else:
        model = joblib.load(model_path)
        if name == "LightGBM":
            model.n_jobs = 1
        evals = evaluate_model(name, model, X_test, y_test, is_pytorch=False)
        trained_models[name] = (model, evals)
        
    results.append({
        "Model": name,
        **{k: v for k, v in evals.items() if k not in ["probs", "preds"]}
    })

# 6. Save Tables
df_res = pd.DataFrame(results)
df_res.to_csv(os.path.join(BASE_DIR, "tables", "lycos_results.csv"), index=False)

reports = []
for name, (_, evals) in trained_models.items():
    pred_labels = np.array(CLASSES)[evals["preds"]]
    true_labels = np.array(CLASSES)[y_test]
    report_dict = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
    for cls_name, cls_metrics in report_dict.items():
        if isinstance(cls_metrics, dict):
            reports.append({
                "Model": name,
                "Class": cls_name,
                "Precision": cls_metrics["precision"],
                "Recall": cls_metrics["recall"],
                "F1-Score": cls_metrics["f1-score"],
                "Support": cls_metrics["support"]
            })
pd.DataFrame(reports).to_csv(os.path.join(BASE_DIR, "tables", "lycos_classification_reports.csv"), index=False)

# 7. Generate curves
# ROC Curve
plt.figure(figsize=(8, 6))
for name, (_, evals) in trained_models.items():
    probs = evals["probs"]
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    for i in range(len(CLASSES)):
        y_true_bin = (y_test == i).astype(int)
        y_prob_bin = probs[:, i]
        # Ignore class 3 (PortScan) in curves since its support is 0 in Lycos
        if np.sum(y_true_bin) == 0:
            continue
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true_bin, y_prob_bin)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[0] = 0.0
    macro_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, label=f"{name} (Macro AUC = {macro_auc:.3f})", lw=2)

plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Macro-Average ROC Curves (Cross-Dataset: Test Lycos 2018)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "figures", "lycos", "roc_curve.png"), dpi=300)
plt.close()

logger.info("Lycos Validation completed successfully!")
