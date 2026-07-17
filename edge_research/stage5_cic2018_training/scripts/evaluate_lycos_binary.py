import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
    precision_recall_curve, auc
)

# 1. Setup Directories & Logging
BASE_DIR = "/Users/harshgoyal/Documents/BTP/BTP/edge_research/stage5_cic2018_training"
TABLE_DIR = os.path.join(BASE_DIR, "tables", "binary_results")
LOG_DIR = os.path.join(BASE_DIR, "logs")

for d in [TABLE_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "binary_lycos_eval_stage5.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("binary_lycos_eval")
logger.info("=" * 70)
logger.info("Stage 5 — Binary LycoS Evaluation (Cross-Dataset)")
logger.info("=" * 70)

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

class MLPClassifierNet(nn.Module):
    def __init__(self, input_dim=82, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

logger.info("Loading preprocessing transformers...")
scaler = joblib.load(os.path.join(BASE_DIR, "artifacts", "scaler.pkl"))
engineered_scaler = joblib.load(os.path.join(BASE_DIR, "artifacts", "engineered_scaler.pkl"))

def apply_feature_engineering(X_scaled_80):
    X_raw = scaler.inverse_transform(X_scaled_80)
    idx_fp = SCALER_FEATURES.index('Total Fwd Packets')
    idx_bp = SCALER_FEATURES.index('Total Backward Packets')
    idx_fl = SCALER_FEATURES.index('Total Length of Fwd Packets')
    idx_bl = SCALER_FEATURES.index('Total Length of Bwd Packets')
    pkt_ratio = X_raw[:, idx_fp] / (X_raw[:, idx_bp] + 1.0)
    len_ratio = X_raw[:, idx_fl] / (X_raw[:, idx_bl] + 1.0)
    eng_scaled = engineered_scaler.transform(np.column_stack([pkt_ratio, len_ratio]))
    return np.column_stack([X_scaled_80, eng_scaled]).astype(np.float32)

logger.info("Loading LycoS Test Set (Cross-Dataset)...")
data_lycos = np.load(os.path.join(BASE_DIR, "artifacts", "lycos_processed.npz"))
X_lycos_80 = data_lycos['X_test'].astype(np.float32)
y_lycos_mc = data_lycos['y_test'].astype(np.int64)
X_lycos_82 = apply_feature_engineering(X_lycos_80)

# Convert to Binary
y_lycos_bin = (y_lycos_mc != 0).astype(int)

def evaluate_binary_model(name, model, X, y_true_bin, is_pytorch=False, device='cpu'):
    logger.info(f"Evaluating {name}...")
    t0 = time.time()
    if is_pytorch:
        model.eval()
        with torch.no_grad():
            outputs = model(torch.tensor(X, dtype=torch.float32).to(device))
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            y_prob_bin = 1.0 - probs[:, 0]  # P(Attack) = 1 - P(Benign)
            y_pred_bin = (np.argmax(probs, axis=1) != 0).astype(int)
    else:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
        else:
            dec = model.decision_function(X)
            if len(dec.shape) == 1:
                probs_bin = 1 / (1 + np.exp(-dec))
                probs = np.column_stack([1 - probs_bin, probs_bin])
            else:
                exp_dec = np.exp(dec - np.max(dec, axis=1, keepdims=True))
                probs = exp_dec / np.sum(exp_dec, axis=1, keepdims=True)
                
        if probs.shape[1] == 2:
            y_prob_bin = probs[:, 1]
        else:
            y_prob_bin = 1.0 - probs[:, 0]
            
        y_pred = model.predict(X)
        y_pred_bin = (y_pred != 0).astype(int)
        
    inference_time = time.time() - t0
    
    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    macro_f1 = f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true_bin, y_pred_bin, average='weighted', zero_division=0)
    bal_acc = balanced_accuracy_score(y_true_bin, y_pred_bin)
    mcc = matthews_corrcoef(y_true_bin, y_pred_bin)
    
    try:
        roc_auc = roc_auc_score(y_true_bin, y_prob_bin)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true_bin, y_prob_bin)
        pr_auc = auc(recall_curve, precision_curve)
    except ValueError:
        roc_auc = None
        pr_auc = None
        
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
        "Inference Time (s)": inference_time
    }

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

results = []

for name, filename, is_pytorch in models_list:
    model_path = os.path.join(BASE_DIR, "models", filename)
    if not os.path.exists(model_path):
        logger.warning(f"Model {filename} not found.")
        continue
        
    if is_pytorch:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = MLPClassifierNet(input_dim=82, num_classes=4).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        res = evaluate_binary_model(name, model, X_lycos_82, y_lycos_bin, is_pytorch=True, device=device)
    else:
        model = joblib.load(model_path)
        if name == "LightGBM":
            model.n_jobs = 1
        res = evaluate_binary_model(name, model, X_lycos_82, y_lycos_bin, is_pytorch=False)
        
    results.append({"Model": name, **res})

df_res = pd.DataFrame(results)
df_res.to_csv(os.path.join(TABLE_DIR, "binary_lycos_ext_results.csv"), index=False)
logger.info("Evaluation complete. Results saved to binary_lycos_ext_results.csv.")
