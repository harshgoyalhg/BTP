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
import seaborn as sns

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix, classification_report,
    roc_curve
)
from sklearn.model_selection import train_test_split

# 1. Setup Directories & Logging
BASE_DIR = "/Users/harshgoyal/Documents/BTP/BTP/edge_research/stage5_cic2018_training"
TABLE_DIR = os.path.join(BASE_DIR, "tables", "binary_results")
FIG_DIR = os.path.join(BASE_DIR, "figures", "binary_results")
LOG_DIR = os.path.join(BASE_DIR, "logs")

for d in [TABLE_DIR, FIG_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "binary_eval_stage5.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("binary_eval")
logger.info("=" * 70)
logger.info("Stage 5 — Binary Evaluation (Within-Dataset and Cross-Dataset)")
logger.info("=" * 70)

CLASSES = ['BENIGN', 'Bot', 'DDoS', 'PortScan']
NUM_CLASSES = 4
INPUT_DIM = 82

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

# 2. Load Transformers
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

# 3. Load Datasets
logger.info("Loading CICIDS-2018 Validation Set...")
data_18 = np.load(os.path.join(BASE_DIR, "artifacts", "cicids2018_processed.npz"), allow_pickle=True)
X_val_80 = data_18['X_val'].astype(np.float32)
y_val = data_18['y_val'].astype(np.int64)

mask = (y_val != 3)
X_val_80 = X_val_80[mask]
y_val = y_val[mask]

logger.info("Loading CICIDS-2017 Dataset to inject PortScan...")
data_17 = np.load(os.path.join(BASE_DIR, "artifacts", "cicids2017_processed.npz"))
ps_idx = np.where(data_17['y_train'] == 3)[0]
np.random.seed(42)
sel = np.random.choice(ps_idx, size=10000, replace=False)
X_ps = data_17['X_train'][sel]
y_ps = data_17['y_train'][sel]

_, X_ps_val, _, y_ps_val = train_test_split(X_ps, y_ps, test_size=0.2, random_state=42, stratify=y_ps)
X_val_80 = np.vstack([X_val_80, X_ps_val])
y_val = np.concatenate([y_val, y_ps_val])
X_val_82 = apply_feature_engineering(X_val_80)
logger.info(f"CIC18 Validation shape (Within-Dataset): {X_val_82.shape}")

logger.info("Loading CICIDS-2017 Test Set (Cross-Dataset)...")
X_cic17_80 = data_17['X_test'].astype(np.float32)
y_cic17 = data_17['y_test'].astype(np.int64)
X_cic17_82 = apply_feature_engineering(X_cic17_80)
logger.info(f"CIC17 Test shape (Cross-Dataset): {X_cic17_82.shape}")

# 4. Evaluation Function
def evaluate_model_binary(name, model_obj, X_eval, y_eval, is_pytorch=False, device='cpu'):
    t0 = time.time()
    
    if is_pytorch:
        model_obj.eval()
        Xt = torch.tensor(X_eval, dtype=torch.float32).to(device)
        with torch.no_grad():
            out = model_obj(Xt)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            y_pred = np.argmax(probs, axis=1)
    else:
        y_pred = model_obj.predict(X_eval)
        if hasattr(model_obj, "predict_proba"):
            probs = model_obj.predict_proba(X_eval)
        else:
            dec = model_obj.decision_function(X_eval)
            if len(dec.shape) == 1:
                p = 1 / (1 + np.exp(-dec))
                probs = np.column_stack([1 - p, p])
            else:
                ed = np.exp(dec - np.max(dec, axis=1, keepdims=True))
                probs = ed / ed.sum(axis=1, keepdims=True)

    y_eval_bin = (y_eval != 0).astype(int)
    y_pred_bin = (y_pred != 0).astype(int)
    probs_bin = 1.0 - probs[:, 0]
    
    inf_time = time.time() - t0

    acc = accuracy_score(y_eval_bin, y_pred_bin)
    prec = precision_score(y_eval_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_eval_bin, y_pred_bin, zero_division=0)
    macro_f1 = f1_score(y_eval_bin, y_pred_bin, zero_division=0)
    wt_f1 = f1_score(y_eval_bin, y_pred_bin, average='weighted', zero_division=0)
    bal_acc = balanced_accuracy_score(y_eval_bin, y_pred_bin)
    mcc = matthews_corrcoef(y_eval_bin, y_pred_bin)

    try:
        roc_auc = roc_auc_score(y_eval_bin, probs_bin)
    except Exception:
        roc_auc = np.nan

    try:
        pv, rv, _ = precision_recall_curve(y_eval_bin, probs_bin)
        pr_auc = auc(rv, pv)
    except Exception:
        pr_auc = np.nan

    logger.info(f"  {name} | Acc={acc:.4f} | F1={macro_f1:.4f} | ROC-AUC={roc_auc:.4f} | Inf={inf_time:.3f}s")

    return {
        "Accuracy": acc, "Precision": prec, "Recall": rec,
        "Macro F1": macro_f1, "Weighted F1": wt_f1,
        "Balanced Accuracy": bal_acc, "MCC": mcc,
        "ROC-AUC": roc_auc, "PR-AUC": pr_auc,
        "Inference Time (s)": inf_time,
        "probs": probs_bin, "preds": y_pred_bin
    }

# 5. Load Models & Evaluate
models_to_eval = [
    ("Logistic Regression", "logistic_regression.pkl", False),
    ("Random Forest", "random_forest.pkl", False),
    ("Extra Trees", "extra_trees.pkl", False),
    ("XGBoost", "xgboost.pkl", False),
    ("LightGBM", "lightgbm.pkl", False),
    ("MLP (PyTorch)", "mlp.pt", True),
    ("SVM (Linear)", "svm_linear.pkl", False),
]

within_results = []
cross_results = []
within_evals = {}
cross_evals = {}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

for name, fname, is_pt in models_to_eval:
    mp = os.path.join(BASE_DIR, "models", fname)
    if not os.path.exists(mp):
        logger.error(f"Model not found: {mp} — skipping.")
        continue

    if is_pt:
        model = MLPClassifierNet(input_dim=INPUT_DIM, num_classes=NUM_CLASSES).to(device)
        model.load_state_dict(torch.load(mp, map_location=device, weights_only=True))
    else:
        model = joblib.load(mp)
        if name == "LightGBM":
            model.n_jobs = 1

    # Evaluate Within-Dataset
    logger.info(f"Evaluating {name} on CIC2018 (Within-Dataset)...")
    ev_w = evaluate_model_binary(name, model, X_val_82, y_val, is_pytorch=is_pt, device=device)
    within_evals[name] = ev_w
    within_results.append({"Model": name, **{k: v for k, v in ev_w.items() if k not in ["probs", "preds"]}})

    # Evaluate Cross-Dataset
    logger.info(f"Evaluating {name} on CIC2017 (Cross-Dataset)...")
    ev_c = evaluate_model_binary(name, model, X_cic17_82, y_cic17, is_pytorch=is_pt, device=device)
    cross_evals[name] = ev_c
    cross_results.append({"Model": name, **{k: v for k, v in ev_c.items() if k not in ["probs", "preds"]}})

# 6. Save Tables
pd.DataFrame(within_results).to_csv(os.path.join(TABLE_DIR, "binary_within_dataset_2018.csv"), index=False)
pd.DataFrame(cross_results).to_csv(os.path.join(TABLE_DIR, "binary_cross_dataset_2017.csv"), index=False)
logger.info("Saved binary result CSVs.")

# 7. Generate Figures (Confusion Matrices & ROC Curves)
def save_confusion_matrix(evals, y_true, title_suffix, fname_prefix):
    y_true_bin = (y_true != 0).astype(int)
    for mname, ev in evals.items():
        cm = confusion_matrix(y_true_bin, ev["preds"])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                    xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
        plt.title(f"{mname} - {title_suffix}")
        plt.ylabel('Actual'); plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"{fname_prefix}_{mname.lower().replace(' ', '_')}.png"), dpi=300)
        plt.close()

def save_roc_curves(evals, y_true, title, fname):
    plt.figure(figsize=(8, 6))
    y_true_bin = (y_true != 0).astype(int)
    for mname, ev in evals.items():
        try:
            fpr, tpr, _ = roc_curve(y_true_bin, ev["probs"])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{mname} (AUC={roc_auc:.3f})", lw=2)
        except Exception:
            pass
    plt.plot([0,1],[0,1],'k--',lw=1.5)
    plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right"); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=300)
    plt.close()

logger.info("Generating figures...")
save_confusion_matrix(within_evals, y_val, "Binary Within-Dataset (CIC2018)", "cm_within")
save_confusion_matrix(cross_evals, y_cic17, "Binary Cross-Dataset (CIC2017)", "cm_cross")
save_roc_curves(within_evals, y_val, "Binary Within-Dataset ROC Curves (CIC2018)", "roc_within.png")
save_roc_curves(cross_evals, y_cic17, "Binary Cross-Dataset ROC Curves (CIC2017)", "roc_cross.png")

logger.info("=" * 70)
logger.info("Binary Evaluation COMPLETE.")
logger.info("=" * 70)
