"""
ensemble_evaluate.py — Stage 6: Soft-Voting Ensemble Evaluation
================================================================
Ensemble: Extra Trees + Random Forest + LightGBM + MLP (PyTorch)
Training source: CICIDS-2017 (Stage 6 trained models)
Strategy: Soft Voting — averaged class probabilities (equal weights)

Evaluation sets:
  1. CIC17 Val     — CICIDS-2017 validation (own dataset)
  2. CIC18 Cross   — CICIDS-2018 cross-dataset (re-scaled by 2017 scaler)
  3. LycoS Ext     — LycoS-Unicas-IDS2018 external (re-scaled by 2017 scaler)

Outputs
-------
models/
    ensemble_config.pkl
tables/
    ensemble_cic17_val_results.csv
    ensemble_cic18_cross_results.csv
    ensemble_lycos_ext_results.csv
    ensemble_all_datasets_summary.csv
    ensemble_cic17_val_class_report.csv
    ensemble_cic18_cross_class_report.csv
    ensemble_lycos_ext_class_report.csv
figures/ensemble/
    confusion_matrix_*.png
    roc_curve_*.png
    pr_curve_*.png
    metrics_bar_comparison.png
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix, classification_report, roc_curve
)

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    BASE_DIR, ARTIFACT_DIR, MODEL_DIR, TABLE_DIR, LOG_DIR, FIG_DIR,
    STAGE5_ARTIFACT_DIR, SCALER_FEATURES, CLASSES, NUM_CLASSES,
    IDX_FWD_PKTS, IDX_BWD_PKTS, IDX_FWD_LEN, IDX_BWD_LEN, RANDOM_STATE
)

# ── Setup ─────────────────────────────────────────────────────────────────────
ENS_FIG_DIR = os.path.join(FIG_DIR, "ensemble")
for d in [ENS_FIG_DIR, TABLE_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "ensemble_evaluation.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("stage6_ensemble")
logger.info("=" * 70)
logger.info("Stage 6 — Ensemble Evaluation (ET + RF + LightGBM + MLP)")
logger.info("=" * 70)

# ── MLP Architecture ──────────────────────────────────────────────────────────
class MLPClassifierNet(nn.Module):
    def __init__(self, input_dim: int = 82, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

INPUT_DIM = 82  # 80 base + 2 engineered

# ── 1. Load Transformers ──────────────────────────────────────────────────────
logger.info("Loading transformers...")
scaler_2017     = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
eng_scaler_2017 = joblib.load(os.path.join(ARTIFACT_DIR, "engineered_scaler.pkl"))
le              = joblib.load(os.path.join(ARTIFACT_DIR, "label_encoder.pkl"))
scaler_2018     = joblib.load(os.path.join(STAGE5_ARTIFACT_DIR, "scaler.pkl"))
logger.info(f"Classes: {le.classes_}")

# ── 2. Feature Engineering Helper ────────────────────────────────────────────
def apply_fe_2017(X_scaled_80: np.ndarray) -> np.ndarray:
    """Inverse-transform with 2017 scaler → ratios → re-scale → append → 82-dim."""
    X_raw = scaler_2017.inverse_transform(X_scaled_80)
    pkt_ratio = X_raw[:, IDX_FWD_PKTS] / (X_raw[:, IDX_BWD_PKTS] + 1.0)
    len_ratio = X_raw[:, IDX_FWD_LEN]  / (X_raw[:, IDX_BWD_LEN]  + 1.0)
    eng = eng_scaler_2017.transform(np.column_stack([pkt_ratio, len_ratio]))
    return np.column_stack([X_scaled_80, eng]).astype(np.float32)

def rescale_from_2018_to_2017(X_scaled_by_2018: np.ndarray) -> np.ndarray:
    """
    X is currently in 2018-scaler space.
    Inverse-transform → apply 2017 scaler → apply 2017 FE → 82-dim.
    """
    X_raw = scaler_2018.inverse_transform(X_scaled_by_2018)
    X_2017_scaled = scaler_2017.transform(X_raw).astype(np.float32)
    pkt_ratio = X_raw[:, IDX_FWD_PKTS] / (X_raw[:, IDX_BWD_PKTS] + 1.0)
    len_ratio = X_raw[:, IDX_FWD_LEN]  / (X_raw[:, IDX_BWD_LEN]  + 1.0)
    eng = eng_scaler_2017.transform(np.column_stack([pkt_ratio, len_ratio]))
    return np.column_stack([X_2017_scaled, eng]).astype(np.float32)

# ── 3. Load Datasets ──────────────────────────────────────────────────────────
# 3a. CIC17 Validation (own dataset)
logger.info("Loading CIC2017 validation set (stage6 artifacts)...")
data_17 = np.load(os.path.join(ARTIFACT_DIR, "cicids2017_processed.npz"), allow_pickle=True)
X_val17_80 = data_17['X_val'].astype(np.float32)
y_val17    = data_17['y_val'].astype(np.int64)
X_val17_82 = apply_fe_2017(X_val17_80)
logger.info(f"CIC17 val: {X_val17_82.shape}, label dist: {np.bincount(y_val17)}")

# 3b. CIC18 Cross-Dataset (re-scaled by 2017 scaler)
logger.info("Loading CIC2018 data from stage5 artifacts for cross-dataset test...")
data_18 = np.load(os.path.join(STAGE5_ARTIFACT_DIR, "cicids2018_processed.npz"), allow_pickle=True)
X_val18_80 = data_18['X_val'].astype(np.float32)
y_val18    = data_18['y_val'].astype(np.int64)

# Remove dummy PortScan; inject real PortScan from stage5's CIC17 cache
mask18 = (y_val18 != 3)
X_val18_80 = X_val18_80[mask18]
y_val18    = y_val18[mask18]
data_17_s5 = np.load(os.path.join(STAGE5_ARTIFACT_DIR, "cicids2017_processed.npz"))
ps_idx = np.where(data_17_s5['y_train'] == 3)[0]
np.random.seed(RANDOM_STATE)
sel = np.random.choice(ps_idx, size=10000, replace=False)
_, X_ps_v, _, y_ps_v = train_test_split(
    data_17_s5['X_train'][sel], data_17_s5['y_train'][sel],
    test_size=0.2, random_state=RANDOM_STATE, stratify=data_17_s5['y_train'][sel])
X_val18_80 = np.vstack([X_val18_80, X_ps_v])
y_val18    = np.concatenate([y_val18, y_ps_v])

# Re-scale CIC18 data from 2018-scaler space → 2017-scaler space
X_val18_82 = rescale_from_2018_to_2017(X_val18_80)
logger.info(f"CIC18 cross: {X_val18_82.shape}, label dist: {np.bincount(y_val18)}")

# 3c. LycoS External (re-scaled by 2017 scaler)
logger.info("Loading LycoS data from stage5 artifacts for external benchmark...")
data_lycos = np.load(os.path.join(STAGE5_ARTIFACT_DIR, "lycos_processed.npz"))
X_lycos_80 = data_lycos['X_test'].astype(np.float32)
y_lycos    = data_lycos['y_test'].astype(np.int64)
X_lycos_82 = rescale_from_2018_to_2017(X_lycos_80)
logger.info(f"LycoS ext: {X_lycos_82.shape}, label dist: {np.bincount(y_lycos)}")

# ── 4. Load Ensemble Members ──────────────────────────────────────────────────
logger.info("Loading ensemble members (ET, RF, LightGBM, MLP)...")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

et_model   = joblib.load(os.path.join(MODEL_DIR, "extra_trees.pkl"))
rf_model   = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
lgbm_model = joblib.load(os.path.join(MODEL_DIR, "lightgbm.pkl"))
lgbm_model.n_jobs = 1

mlp_model = MLPClassifierNet(input_dim=INPUT_DIM, num_classes=NUM_CLASSES).to(device)
mlp_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "mlp.pt"),
                                     map_location=device, weights_only=True))
mlp_model.eval()
logger.info("All 4 ensemble members loaded.")

# Save ensemble config
ensemble_cfg = {
    'members': ['Extra Trees', 'Random Forest', 'LightGBM', 'MLP (PyTorch)'],
    'strategy': 'Soft Voting (equal weights, averaged probabilities)',
    'weights': [0.25, 0.25, 0.25, 0.25],
    'trained_on': 'CICIDS-2017',
    'input_dim': INPUT_DIM,
    'num_classes': NUM_CLASSES,
    'classes': CLASSES,
}
joblib.dump(ensemble_cfg, os.path.join(MODEL_DIR, "ensemble_config.pkl"))
logger.info("Saved ensemble_config.pkl")

# ── 5. Soft Voting Inference ──────────────────────────────────────────────────
def get_proba_sklearn(model, X):
    return model.predict_proba(X)

def get_proba_mlp(model, X, dev):
    model.eval()
    t = torch.tensor(X, dtype=torch.float32).to(dev)
    with torch.no_grad():
        out = model(t)
        return torch.softmax(out, dim=1).cpu().numpy()

def ensemble_predict(X_82: np.ndarray):
    p_et   = get_proba_sklearn(et_model,   X_82)
    p_rf   = get_proba_sklearn(rf_model,   X_82)
    p_lgbm = get_proba_sklearn(lgbm_model, X_82)
    p_mlp  = get_proba_mlp(mlp_model, X_82, device)
    avg_p  = (p_et + p_rf + p_lgbm + p_mlp) / 4.0
    return avg_p, np.argmax(avg_p, axis=1)

# ── 6. Evaluation Function ────────────────────────────────────────────────────
def evaluate_ensemble(X, y, dataset_label, active_classes=None):
    logger.info(f"\n--- Evaluating Ensemble on: {dataset_label} ---")
    t0 = time.time()
    probs, y_pred = ensemble_predict(X)
    inf_time = time.time() - t0

    acc     = accuracy_score(y, y_pred)
    prec    = precision_score(y, y_pred, average='weighted', zero_division=0)
    rec     = recall_score(y, y_pred, average='weighted', zero_division=0)
    mac_f1  = f1_score(y, y_pred, average='macro', zero_division=0)
    wt_f1   = f1_score(y, y_pred, average='weighted', zero_division=0)
    bal_acc = balanced_accuracy_score(y, y_pred)
    mcc     = matthews_corrcoef(y, y_pred)

    try:
        roc_auc = roc_auc_score(y, probs, multi_class='ovr', average='macro')
    except Exception as e:
        logger.warning(f"ROC-AUC failed: {e}"); roc_auc = np.nan

    use_cls = active_classes if active_classes else list(range(NUM_CLASSES))
    pr_list = []
    for i in use_cls:
        try:
            yb = (y == i).astype(int)
            pv, rv, _ = precision_recall_curve(yb, probs[:, i])
            pr_list.append(auc(rv, pv))
        except Exception:
            pass
    pr_auc = np.mean(pr_list) if pr_list else np.nan

    roc_str = f"{roc_auc:.4f}" if not np.isnan(roc_auc) else "N/A"
    pr_str = f"{pr_auc:.4f}" if not np.isnan(pr_auc) else "N/A"
    strategy = 'Soft Voting'
    logger.info(f"  Ensemble ({strategy}): Acc={acc:.4f} | Wt-F1={wt_f1:.4f} "
                f"| Macro-F1={mac_f1:.4f} | MCC={mcc:.4f} | Inf={inf_time:.3f}s "
                f"| ROC-AUC={roc_str} | PR-AUC={pr_str}")

    return {
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
        'PR-AUC':  round(pr_auc  * 100, 4) if not np.isnan(pr_auc)  else 'N/A',
        'Inference Time (s)': round(inf_time, 4),
        'N Samples': len(y),
    }, probs, y_pred

# ── 7. Figure Helpers ─────────────────────────────────────────────────────────
def save_confusion_matrix(y_true, y_pred, dataset_label, filename, cmap='Blues'):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
                linewidths=0.5, linecolor='white')
    ax.set_title(f"Ensemble Confusion Matrix\n{dataset_label}", fontsize=13, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=11); ax.set_xlabel('Predicted', fontsize=11)
    plt.tight_layout()
    path = os.path.join(ENS_FIG_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close()
    logger.info(f"Saved: {path}")

def save_roc(y_true, probs, dataset_label, filename, active_classes=None):
    use_cls = active_classes if active_classes else list(range(NUM_CLASSES))
    fig, ax = plt.subplots(figsize=(8, 6))
    mean_fpr = np.linspace(0, 1, 200)
    tprs = []
    for i in use_cls:
        yb = (y_true == i).astype(int)
        if yb.sum() == 0: continue
        fpr_, tpr_, _ = roc_curve(yb, probs[:, i])
        tprs.append(np.interp(mean_fpr, fpr_, tpr_))
        ax.plot(mean_fpr, tprs[-1], alpha=0.4, lw=1.2,
                label=f"{CLASSES[i]} (AUC={auc(fpr_, tpr_):.3f})")
    if tprs:
        mean_tpr = np.mean(tprs, axis=0); mean_tpr[0] = 0.0
        macro_a = auc(mean_fpr, mean_tpr)
        ax.plot(mean_fpr, mean_tpr, 'k-', lw=2.5, label=f"Macro (AUC={macro_a:.3f})")
    ax.plot([0,1],[0,1],'r--',lw=1.2,label='Random')
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f"Ensemble ROC Curves\n{dataset_label}", fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(ENS_FIG_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close()
    logger.info(f"Saved: {path}")

def save_pr(y_true, probs, dataset_label, filename, active_classes=None):
    use_cls = active_classes if active_classes else list(range(NUM_CLASSES))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in use_cls:
        yb = (y_true == i).astype(int)
        if yb.sum() == 0: continue
        pv, rv, _ = precision_recall_curve(yb, probs[:, i])
        ax.plot(rv, pv, lw=1.5, label=f"{CLASSES[i]} (AUC={auc(rv, pv):.3f})")
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    ax.set_xlabel('Recall', fontsize=11); ax.set_ylabel('Precision', fontsize=11)
    ax.set_title(f"Ensemble PR Curves\n{dataset_label}", fontsize=13, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(ENS_FIG_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close()
    logger.info(f"Saved: {path}")

# ── 8. Run Evaluations ────────────────────────────────────────────────────────
# 8a. CIC17 Validation (own dataset)
logger.info("\n" + "=" * 60)
logger.info("EVALUATION 1: CIC2017 Validation (Own Dataset)")
logger.info("=" * 60)
m17, p17, pred17 = evaluate_ensemble(X_val17_82, y_val17, "CIC2017 Validation (Own Dataset)")
pd.DataFrame([m17]).to_csv(os.path.join(TABLE_DIR, "ensemble_cic17_val_results.csv"), index=False)
rpt17 = classification_report(np.array(CLASSES)[y_val17], np.array(CLASSES)[pred17],
                               output_dict=True, zero_division=0)
pd.DataFrame(rpt17).T.to_csv(os.path.join(TABLE_DIR, "ensemble_cic17_val_class_report.csv"))
save_confusion_matrix(y_val17, pred17, "CIC2017 Validation Set", "confusion_matrix_cic17_val.png", 'Blues')
save_roc(y_val17, p17, "CIC2017 Validation Set", "roc_curve_cic17_val.png")
save_pr(y_val17, p17, "CIC2017 Validation Set", "pr_curve_cic17_val.png")

# 8b. CIC18 Cross-Dataset
logger.info("\n" + "=" * 60)
logger.info("EVALUATION 2: CIC2018 Cross-Dataset")
logger.info("=" * 60)
m18, p18, pred18 = evaluate_ensemble(X_val18_82, y_val18, "CICIDS2018 Cross-Dataset")
pd.DataFrame([m18]).to_csv(os.path.join(TABLE_DIR, "ensemble_cic18_cross_results.csv"), index=False)
rpt18 = classification_report(np.array(CLASSES)[y_val18], np.array(CLASSES)[pred18],
                               output_dict=True, zero_division=0)
pd.DataFrame(rpt18).T.to_csv(os.path.join(TABLE_DIR, "ensemble_cic18_cross_class_report.csv"))
save_confusion_matrix(y_val18, pred18, "CICIDS2018 Cross-Dataset", "confusion_matrix_cic18_cross.png", 'Oranges')
save_roc(y_val18, p18, "CICIDS2018 Cross-Dataset", "roc_curve_cic18_cross.png")
save_pr(y_val18, p18, "CICIDS2018 Cross-Dataset", "pr_curve_cic18_cross.png")

# 8c. LycoS External (PortScan absent)
logger.info("\n" + "=" * 60)
logger.info("EVALUATION 3: LycoS External")
logger.info("=" * 60)
lycos_active = [0, 1, 2]  # PortScan (3) absent in LycoS
ml, pl, predl = evaluate_ensemble(X_lycos_82, y_lycos,
                                   "LycoS-Unicas-IDS2018 External", active_classes=lycos_active)
pd.DataFrame([ml]).to_csv(os.path.join(TABLE_DIR, "ensemble_lycos_ext_results.csv"), index=False)
rptl = classification_report(np.array(CLASSES)[y_lycos], np.array(CLASSES)[predl],
                              output_dict=True, zero_division=0)
pd.DataFrame(rptl).T.to_csv(os.path.join(TABLE_DIR, "ensemble_lycos_ext_class_report.csv"))
save_confusion_matrix(y_lycos, predl, "LycoS External", "confusion_matrix_lycos_ext.png", 'Purples')
save_roc(y_lycos, pl, "LycoS External", "roc_curve_lycos_ext.png", active_classes=lycos_active)
save_pr(y_lycos, pl, "LycoS External", "pr_curve_lycos_ext.png", active_classes=lycos_active)

# ── 9. Combined Summary ───────────────────────────────────────────────────────
df_summary = pd.DataFrame([m17, m18, ml])
df_summary.to_csv(os.path.join(TABLE_DIR, "ensemble_all_datasets_summary.csv"), index=False)
logger.info("Saved: ensemble_all_datasets_summary.csv")

cols_disp = ['Dataset', 'Accuracy', 'Macro F1', 'Weighted F1', 'Balanced Accuracy', 'MCC', 'ROC-AUC', 'PR-AUC']
print("\n" + "=" * 80)
print("ENSEMBLE RESULTS — ALL DATASETS")
print("=" * 80)
print(df_summary[cols_disp].to_string(index=False))

# ── 10. Bar Chart ─────────────────────────────────────────────────────────────
metric_cols = ['Accuracy', 'Macro F1', 'Weighted F1', 'Balanced Accuracy', 'MCC']
datasets    = ['CIC17 Val', 'CIC18 Cross', 'LycoS Ext']
values = [[m17[c] for c in metric_cols],
          [m18[c] for c in metric_cols],
          [ml[c]  for c in metric_cols]]

x = np.arange(len(metric_cols)); width = 0.25
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
ax.set_title('Ensemble Model (ET + RF + LightGBM + MLP)\n'
             'Performance Across All Datasets — CICIDS-2017 Trained',
             fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim(0, 108); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
path = os.path.join(ENS_FIG_DIR, "metrics_bar_comparison.png")
plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close()
logger.info(f"Saved: {path}")

logger.info("=" * 70)
logger.info("Stage 6 Ensemble Evaluation COMPLETE.")
logger.info("=" * 70)
