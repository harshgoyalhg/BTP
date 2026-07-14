"""
Ensemble Evaluation Script
===========================
Ensemble Strategy : Soft Voting (averaged class probabilities)
Members           : Extra Trees, Random Forest, LightGBM, MLP (PyTorch)
Training Source   : CSE-CIC-IDS2018 (pre-trained models loaded from disk)
Evaluation Sets   :
    1. CIC18 Val     (internal validation)
    2. CIC17 Cross   (CICIDS2017 cross-dataset)
    3. LycoS Ext     (LycoS-Unicas-IDS2018 external)

Outputs
-------
models/
    ensemble_config.pkl              <- member names + weights
tables/
    ensemble_cic18_val_results.csv   <- internal validation metrics
    ensemble_cic17_cross_results.csv <- CICIDS2017 cross-dataset metrics
    ensemble_lycos_ext_results.csv   <- LycoS external metrics
    ensemble_all_datasets_summary.csv<- combined 3-row summary
figures/ensemble/
    confusion_matrix_cic18_val.png
    confusion_matrix_cic17_cross.png
    confusion_matrix_lycos_ext.png
    roc_curve_cic18_val.png
    roc_curve_cic17_cross.png
    roc_curve_lycos_ext.png
    pr_curve_cic18_val.png
    pr_curve_cic17_cross.png
    pr_curve_lycos_ext.png
    metrics_bar_comparison.png       <- all 3 datasets side-by-side
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

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

# ─── 1. Paths & Directories ────────────────────────────────────────────────────
BASE_DIR    = "/Users/harshgoyal/Documents/BTP/BTP/edge_research/stage5_cic2018_training"
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
TABLE_DIR   = os.path.join(BASE_DIR, "tables")
FIG_DIR     = os.path.join(BASE_DIR, "figures", "ensemble")

for d in [TABLE_DIR, FIG_DIR, os.path.join(BASE_DIR, "logs")]:
    os.makedirs(d, exist_ok=True)

# ─── 2. Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "logs", "ensemble_evaluation.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ensemble_eval")
logger.info("=" * 70)
logger.info("Ensemble Evaluation: ET + RF + LGBM + MLP (Soft Voting)")
logger.info("=" * 70)

# ─── 3. Constants ──────────────────────────────────────────────────────────────
CLASSES = ['BENIGN', 'Bot', 'DDoS', 'PortScan']
NUM_CLASSES = 4
INPUT_DIM = 82  # 80 base + 2 engineered

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

# ─── 4. MLP Architecture (must match train_cic2018.py exactly) ─────────────────
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

# ─── 5. Load Transformers ──────────────────────────────────────────────────────
logger.info("Loading preprocessing transformers (scaler, engineered_scaler, label_encoder)...")
scaler            = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
engineered_scaler = joblib.load(os.path.join(ARTIFACT_DIR, "engineered_scaler.pkl"))
le                = joblib.load(os.path.join(ARTIFACT_DIR, "label_encoder.pkl"))
logger.info(f"Label classes: {le.classes_}")

# ─── 6. Feature Engineering Helper ────────────────────────────────────────────
def apply_feature_engineering(X_scaled_80):
    """Inverse-transform → compute 2 ratios → re-scale → append. Returns (N, 82) array."""
    X_raw = scaler.inverse_transform(X_scaled_80)
    idx_fp = SCALER_FEATURES.index('Total Fwd Packets')
    idx_bp = SCALER_FEATURES.index('Total Backward Packets')
    idx_fl = SCALER_FEATURES.index('Total Length of Fwd Packets')
    idx_bl = SCALER_FEATURES.index('Total Length of Bwd Packets')
    pkt_ratio = X_raw[:, idx_fp] / (X_raw[:, idx_bp] + 1.0)
    len_ratio = X_raw[:, idx_fl] / (X_raw[:, idx_bl] + 1.0)
    eng_scaled = engineered_scaler.transform(np.column_stack([pkt_ratio, len_ratio]))
    return np.column_stack([X_scaled_80, eng_scaled]).astype(np.float32)

# ─── 7. Load Datasets ──────────────────────────────────────────────────────────
logger.info("Loading CIC2018 validation set...")
data_18 = np.load(os.path.join(ARTIFACT_DIR, "cicids2018_processed.npz"), allow_pickle=True)
X_val_80 = data_18['X_val'].astype(np.float32)
y_val    = data_18['y_val'].astype(np.int64)
# Remove dummy PortScan (label=3) from cached val that may have been stored before injection
mask = (y_val != 3)
X_val_80 = X_val_80[mask]
y_val    = y_val[mask]
# Inject real PortScan from CIC17
data_17 = np.load(os.path.join(ARTIFACT_DIR, "cicids2017_processed.npz"))
ps_idx   = np.where(data_17['y_train'] == 3)[0]
np.random.seed(42)
sel      = np.random.choice(ps_idx, size=10000, replace=False)
X_ps     = data_17['X_train'][sel]
y_ps     = data_17['y_train'][sel]
from sklearn.model_selection import train_test_split
_, X_ps_val, _, y_ps_val = train_test_split(X_ps, y_ps, test_size=0.2, random_state=42, stratify=y_ps)
X_val_80 = np.vstack([X_val_80, X_ps_val])
y_val    = np.concatenate([y_val, y_ps_val])
X_val_82 = apply_feature_engineering(X_val_80)
logger.info(f"CIC18 val shape: {X_val_82.shape}, label dist: {np.bincount(y_val)}")

logger.info("Loading CIC2017 cross-dataset test set...")
X_cic17_80 = data_17['X_test'].astype(np.float32)
y_cic17    = data_17['y_test'].astype(np.int64)
X_cic17_82 = apply_feature_engineering(X_cic17_80)
logger.info(f"CIC17 test shape: {X_cic17_82.shape}, label dist: {np.bincount(y_cic17)}")

logger.info("Loading LycoS external test set...")
data_lycos = np.load(os.path.join(ARTIFACT_DIR, "lycos_processed.npz"))
X_lycos_80 = data_lycos['X_test'].astype(np.float32)
y_lycos    = data_lycos['y_test'].astype(np.int64)
X_lycos_82 = apply_feature_engineering(X_lycos_80)
logger.info(f"LycoS test shape: {X_lycos_82.shape}, label dist: {np.bincount(y_lycos)}")

# ─── 8. Load Ensemble Members ──────────────────────────────────────────────────
logger.info("Loading ensemble member models...")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"MPS device: {device}")

et_model   = joblib.load(os.path.join(MODEL_DIR, "extra_trees.pkl"))
rf_model   = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
lgbm_model = joblib.load(os.path.join(MODEL_DIR, "lightgbm.pkl"))
lgbm_model.n_jobs = 1  # prevent OpenMP thread fork issues on macOS

mlp_model = MLPClassifierNet(input_dim=INPUT_DIM, num_classes=NUM_CLASSES).to(device)
mlp_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "mlp.pt"), map_location=device))
mlp_model.eval()
logger.info("All 4 ensemble members loaded: ET, RF, LightGBM, MLP(PyTorch)")

# Save ensemble config
ensemble_config = {
    'members': ['Extra Trees', 'Random Forest', 'LightGBM', 'MLP (PyTorch)'],
    'strategy': 'Soft Voting (equal weights, averaged probabilities)',
    'weights': [0.25, 0.25, 0.25, 0.25],
    'input_dim': INPUT_DIM,
    'num_classes': NUM_CLASSES,
    'classes': CLASSES,
    'trained_on': 'CSE-CIC-IDS2018',
}
joblib.dump(ensemble_config, os.path.join(MODEL_DIR, "ensemble_config.pkl"))
logger.info("Saved ensemble_config.pkl")

# ─── 9. Soft Voting Inference ──────────────────────────────────────────────────
def get_proba_sklearn(model, X):
    return model.predict_proba(X)

def get_proba_mlp(model, X, dev):
    model.eval()
    t = torch.tensor(X, dtype=torch.float32).to(dev)
    with torch.no_grad():
        out = model(t)
        return torch.softmax(out, dim=1).cpu().numpy()

def ensemble_predict(X_82):
    """Soft-voting: average probability matrices from all 4 members."""
    p_et   = get_proba_sklearn(et_model,   X_82)
    p_rf   = get_proba_sklearn(rf_model,   X_82)
    p_lgbm = get_proba_sklearn(lgbm_model, X_82)
    p_mlp  = get_proba_mlp(mlp_model, X_82, device)
    avg_p  = (p_et + p_rf + p_lgbm + p_mlp) / 4.0
    return avg_p, np.argmax(avg_p, axis=1)

# ─── 10. Evaluation Function ───────────────────────────────────────────────────
def evaluate_ensemble(X, y, dataset_label, active_classes=None):
    """
    Evaluates the ensemble on X, y.
    active_classes: list of class indices actually present in y (for LycoS which lacks PortScan)
    """
    logger.info(f"--- Evaluating Ensemble on: {dataset_label} ---")
    t0 = time.time()
    probs, y_pred = ensemble_predict(X)
    inf_time = time.time() - t0

    # Core metrics
    acc      = accuracy_score(y, y_pred)
    prec     = precision_score(y, y_pred, average='weighted', zero_division=0)
    rec      = recall_score(y, y_pred, average='weighted', zero_division=0)
    mac_f1   = f1_score(y, y_pred, average='macro', zero_division=0)
    wt_f1    = f1_score(y, y_pred, average='weighted', zero_division=0)
    bal_acc  = balanced_accuracy_score(y, y_pred)
    mcc      = matthews_corrcoef(y, y_pred)

    # ROC-AUC (OvR macro)
    try:
        roc_auc = roc_auc_score(y, probs, multi_class='ovr', average='macro')
    except Exception as e:
        logger.warning(f"ROC-AUC not computed for {dataset_label}: {e}")
        roc_auc = np.nan

    # PR-AUC per class then average
    pr_aucs = []
    use_classes = active_classes if active_classes else list(range(NUM_CLASSES))
    for i in use_classes:
        try:
            y_bin = (y == i).astype(int)
            p_vals, r_vals, _ = precision_recall_curve(y_bin, probs[:, i])
            pr_aucs.append(auc(r_vals, p_vals))
        except Exception:
            pass
    pr_auc = np.mean(pr_aucs) if pr_aucs else np.nan

    roc_str = f"{roc_auc:.4f}" if not np.isnan(roc_auc) else "N/A"
    logger.info(f"  Accuracy={acc:.4f} | MCC={mcc:.4f} | Macro-F1={mac_f1:.4f} | "
                f"ROC-AUC={roc_str} | Inf={inf_time:.3f}s")

    metrics = {
        'Dataset': dataset_label,
        'Ensemble Members': 'ET + RF + LightGBM + MLP',
        'Strategy': 'Soft Voting (equal weights)',
        'Accuracy': round(acc * 100, 4),
        'Precision (Weighted)': round(prec * 100, 4),
        'Recall (Weighted)': round(rec * 100, 4),
        'Macro F1': round(mac_f1 * 100, 4),
        'Weighted F1': round(wt_f1 * 100, 4),
        'Balanced Accuracy': round(bal_acc * 100, 4),
        'MCC': round(mcc * 100, 4),
        'ROC-AUC': round(roc_auc * 100, 4) if not np.isnan(roc_auc) else 'N/A',
        'PR-AUC': round(pr_auc * 100, 4) if not np.isnan(pr_auc) else 'N/A',
        'Inference Time (s)': round(inf_time, 4),
        'N Samples': len(y),
    }
    return metrics, probs, y_pred

# ─── 11. Figure Helpers ────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, dataset_label, filename, cmap='Blues'):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
                linewidths=0.5, linecolor='white')
    ax.set_title(f"Ensemble Confusion Matrix\n{dataset_label}", fontsize=13, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_xlabel('Predicted', fontsize=11)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")

def plot_roc(y_true, probs, dataset_label, filename, active_classes=None):
    use_classes = active_classes if active_classes else list(range(NUM_CLASSES))
    fig, ax = plt.subplots(figsize=(8, 6))
    mean_fpr = np.linspace(0, 1, 200)
    tprs = []
    for i in use_classes:
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin, probs[:, i])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        cls_auc = auc(fpr, tpr)
        ax.plot(mean_fpr, tprs[-1], alpha=0.4, lw=1.2, label=f"{CLASSES[i]} (AUC={cls_auc:.3f})")
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[0] = 0.0
    macro_auc = auc(mean_fpr, mean_tpr)
    ax.plot(mean_fpr, mean_tpr, 'k-', lw=2.5, label=f"Macro Average (AUC={macro_auc:.3f})")
    ax.plot([0, 1], [0, 1], 'r--', lw=1.2, label='Random Classifier')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f"Ensemble ROC Curves\n{dataset_label}", fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")

def plot_pr(y_true, probs, dataset_label, filename, active_classes=None):
    use_classes = active_classes if active_classes else list(range(NUM_CLASSES))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in use_classes:
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0:
            continue
        prec_vals, rec_vals, _ = precision_recall_curve(y_bin, probs[:, i])
        pr_a = auc(rec_vals, prec_vals)
        ax.plot(rec_vals, prec_vals, lw=1.5, label=f"{CLASSES[i]} (AUC={pr_a:.3f})")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title(f"Ensemble PR Curves\n{dataset_label}", fontsize=13, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")

# ─── 12. Run Evaluations ───────────────────────────────────────────────────────

# 12a. CIC18 Validation (own dataset) — all 4 classes present
logger.info("\n" + "=" * 60)
logger.info("EVALUATION 1: CIC18 Validation Set (Own Dataset)")
logger.info("=" * 60)
m18, p18, pred18 = evaluate_ensemble(X_val_82, y_val, "CIC2018 Validation (Own Dataset)")
pd.DataFrame([m18]).to_csv(os.path.join(TABLE_DIR, "ensemble_cic18_val_results.csv"), index=False)
logger.info("Saved: ensemble_cic18_val_results.csv")

# Per-class classification report
report18 = classification_report(
    np.array(CLASSES)[y_val], np.array(CLASSES)[pred18],
    output_dict=True, zero_division=0
)
df_report18 = pd.DataFrame(report18).T
df_report18.to_csv(os.path.join(TABLE_DIR, "ensemble_cic18_val_class_report.csv"))
logger.info("Saved: ensemble_cic18_val_class_report.csv")

plot_confusion_matrix(y_val, pred18, "CIC2018 Validation Set",
                      "confusion_matrix_cic18_val.png", cmap='Blues')
plot_roc(y_val, p18, "CIC2018 Validation Set", "roc_curve_cic18_val.png")
plot_pr(y_val, p18, "CIC2018 Validation Set", "pr_curve_cic18_val.png")

# 12b. CIC17 Cross-Dataset — all 4 classes present
logger.info("\n" + "=" * 60)
logger.info("EVALUATION 2: CICIDS2017 Cross-Dataset")
logger.info("=" * 60)
m17, p17, pred17 = evaluate_ensemble(X_cic17_82, y_cic17, "CICIDS2017 Cross-Dataset")
pd.DataFrame([m17]).to_csv(os.path.join(TABLE_DIR, "ensemble_cic17_cross_results.csv"), index=False)
logger.info("Saved: ensemble_cic17_cross_results.csv")

report17 = classification_report(
    np.array(CLASSES)[y_cic17], np.array(CLASSES)[pred17],
    output_dict=True, zero_division=0
)
df_report17 = pd.DataFrame(report17).T
df_report17.to_csv(os.path.join(TABLE_DIR, "ensemble_cic17_cross_class_report.csv"))
logger.info("Saved: ensemble_cic17_cross_class_report.csv")

plot_confusion_matrix(y_cic17, pred17, "CICIDS2017 Cross-Dataset",
                      "confusion_matrix_cic17_cross.png", cmap='Oranges')
plot_roc(y_cic17, p17, "CICIDS2017 Cross-Dataset", "roc_curve_cic17_cross.png")
plot_pr(y_cic17, p17, "CICIDS2017 Cross-Dataset", "pr_curve_cic17_cross.png")

# 12c. LycoS External — only classes 0 (BENIGN), 1 (Bot), 2 (DDoS) present
logger.info("\n" + "=" * 60)
logger.info("EVALUATION 3: LycoS External Dataset")
logger.info("=" * 60)
lycos_active = [0, 1, 2]  # PortScan (3) absent in LycoS
ml, pl, predl = evaluate_ensemble(X_lycos_82, y_lycos, "LycoS-Unicas-IDS2018 External",
                                  active_classes=lycos_active)
pd.DataFrame([ml]).to_csv(os.path.join(TABLE_DIR, "ensemble_lycos_ext_results.csv"), index=False)
logger.info("Saved: ensemble_lycos_ext_results.csv")

reportl = classification_report(
    np.array(CLASSES)[y_lycos], np.array(CLASSES)[predl],
    output_dict=True, zero_division=0
)
df_reportl = pd.DataFrame(reportl).T
df_reportl.to_csv(os.path.join(TABLE_DIR, "ensemble_lycos_ext_class_report.csv"))
logger.info("Saved: ensemble_lycos_ext_class_report.csv")

plot_confusion_matrix(y_lycos, predl, "LycoS External Dataset",
                      "confusion_matrix_lycos_ext.png", cmap='Purples')
plot_roc(y_lycos, pl, "LycoS External Dataset", "roc_curve_lycos_ext.png",
         active_classes=lycos_active)
plot_pr(y_lycos, pl, "LycoS External Dataset", "pr_curve_lycos_ext.png",
        active_classes=lycos_active)

# ─── 13. Combined Summary Table ────────────────────────────────────────────────
logger.info("\n" + "=" * 60)
logger.info("COMBINED SUMMARY")
logger.info("=" * 60)
df_summary = pd.DataFrame([m18, m17, ml])
df_summary.to_csv(os.path.join(TABLE_DIR, "ensemble_all_datasets_summary.csv"), index=False)
logger.info("Saved: ensemble_all_datasets_summary.csv")

print("\n" + "=" * 80)
print("ENSEMBLE RESULTS — ALL DATASETS")
print("=" * 80)
cols_display = ['Dataset', 'Accuracy', 'Macro F1', 'Weighted F1',
                'Balanced Accuracy', 'MCC', 'ROC-AUC', 'PR-AUC']
print(df_summary[cols_display].to_string(index=False))

# ─── 14. Bar Chart: All Metrics Across 3 Datasets ─────────────────────────────
metric_cols = ['Accuracy', 'Macro F1', 'Weighted F1', 'Balanced Accuracy', 'MCC']
datasets    = ['CIC18 Val', 'CIC17 Cross', 'LycoS Ext']
values      = [
    [m18[c] for c in metric_cols],
    [m17[c] for c in metric_cols],
    [ml[c]  for c in metric_cols],
]

x = np.arange(len(metric_cols))
width = 0.25
fig, ax = plt.subplots(figsize=(12, 6))
colors = ['#4C72B0', '#DD8452', '#55A868']
for i, (ds, vals, col) in enumerate(zip(datasets, values, colors)):
    bars = ax.bar(x + i * width, vals, width, label=ds, color=col, alpha=0.87)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6,
                f'{val:.1f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

ax.set_xticks(x + width)
ax.set_xticklabels(metric_cols, fontsize=10)
ax.set_ylabel('Score (%)', fontsize=11)
ax.set_title('Ensemble Model (ET + RF + LightGBM + MLP)\nPerformance Across All Datasets',
             fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim(0, 108)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
path = os.path.join(FIG_DIR, "metrics_bar_comparison.png")
plt.savefig(path, dpi=300, bbox_inches='tight')
plt.close()
logger.info(f"Saved: {path}")

logger.info("=" * 70)
logger.info("Ensemble Evaluation COMPLETE.")
logger.info("=" * 70)
