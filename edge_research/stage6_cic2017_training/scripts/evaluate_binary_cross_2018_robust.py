"""
evaluate_cross_2018.py — Stage 6: Cross-Dataset Evaluation
===========================================================
Direction: Train on CICIDS-2017  →  Test on CICIDS-2018

Methodology:
  1. Load stage5 cached CICIDS-2018 validation set (X_val scaled by 2018 scaler)
  2. Inverse-transform using stage5 scaler → recover raw feature values
  3. Apply stage6 (2017) scaler → transform into 2017 feature space
  4. Apply stage6 engineered_scaler → append 2 ratio features (82-dim)
  5. Run all 8 stage6 models trained on CICIDS-2017

This is the counterpart to stage5/evaluate_cross_dataset.py which goes
Train-2018 → Test-2017. Together they form a complete bidirectional
cross-dataset evaluation suite for the research paper.
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix, classification_report,
    roc_curve
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    BASE_DIR, ARTIFACT_DIR, MODEL_DIR, TABLE_DIR, LOG_DIR, FIG_DIR,
    STAGE5_ARTIFACT_DIR, SCALER_FEATURES, CLASSES, NUM_CLASSES,
    IDX_FWD_PKTS, IDX_BWD_PKTS, IDX_FWD_LEN, IDX_BWD_LEN, RANDOM_STATE
)

# ── Setup ─────────────────────────────────────────────────────────────────────
CROSS_FIG_DIR = os.path.join(FIG_DIR, "cross_2018")
for d in [CROSS_FIG_DIR, TABLE_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "cross_2018_eval.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("cross_2018_eval")
logger.info("=" * 70)
logger.info("Stage 6 — Cross-Dataset Evaluation: Train-CIC2017 → Test-CIC2018")
logger.info("=" * 70)

# ── MLP Architecture (must match train_cic2017.py exactly) ───────────────────
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

# ── 1. Load Transformers ──────────────────────────────────────────────────────
logger.info("Loading Stage 5 (2018) scaler for inverse-transform...")
scaler_2018 = joblib.load(os.path.join(STAGE5_ARTIFACT_DIR, "scaler.pkl"))

logger.info("Loading Stage 6 (2017) scaler and engineered_scaler...")
scaler_2017      = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
eng_scaler_2017 = joblib.load(os.path.join(ARTIFACT_DIR, "engineered_scaler_robust.pkl"))
le               = joblib.load(os.path.join(ARTIFACT_DIR, "label_encoder.pkl"))
logger.info(f"Classes: {le.classes_}")

# ── 2. Load & Prepare CICIDS-2018 Test Set ───────────────────────────────────
logger.info("Loading CICIDS-2018 cached data from stage5 artifacts...")
data_18 = np.load(os.path.join(STAGE5_ARTIFACT_DIR, "cicids2018_processed.npz"), allow_pickle=True)
X_val_18_scaled_by_2018 = data_18['X_val'].astype(np.float32)
y_val_18                 = data_18['y_val'].astype(np.int64)

# Remove dummy PortScan rows (label=3) that may exist in cached data
mask = (y_val_18 != 3)
X_val_18_scaled_by_2018 = X_val_18_scaled_by_2018[mask]
y_val_18                 = y_val_18[mask]

# Inject real PortScan from 2017 (same injection as stage5 used for its val set)
logger.info("Injecting real PortScan samples into CIC2018 val set...")
data_17_s5 = np.load(os.path.join(STAGE5_ARTIFACT_DIR, "cicids2017_processed.npz"))
ps_idx = np.where(data_17_s5['y_train'] == 3)[0]
np.random.seed(RANDOM_STATE)
sel = np.random.choice(ps_idx, size=10000, replace=False)
X_ps_17 = data_17_s5['X_train'][sel]       # scaled by 2018 scaler
y_ps_17 = data_17_s5['y_train'][sel]
_, X_ps_val, _, y_ps_val = train_test_split(
    X_ps_17, y_ps_17, test_size=0.2, random_state=RANDOM_STATE, stratify=y_ps_17)
X_val_18_scaled_by_2018 = np.vstack([X_val_18_scaled_by_2018, X_ps_val])
y_val_18                 = np.concatenate([y_val_18, y_ps_val])
logger.info(f"CIC2018 test set shape (after PS injection): {X_val_18_scaled_by_2018.shape}")
logger.info(f"Label distribution: {np.bincount(y_val_18)}")

# ── 3. Re-scale CIC2018 Data into 2017 Feature Space ─────────────────────────
logger.info("Inverse-transforming CIC2018 features from 2018 scaler space...")
X_raw_18 = scaler_2018.inverse_transform(X_val_18_scaled_by_2018)

logger.info("Applying Stage 6 (2017) scaler to CIC2018 raw features...")
X_18_scaled_by_2017 = scaler_2017.transform(X_raw_18).astype(np.float32)

logger.info("Applying feature engineering (2017 engineered_scaler_robust)...")
pkt_ratio = X_raw_18[:, IDX_FWD_PKTS] / (X_raw_18[:, IDX_BWD_PKTS] + 1.0)
len_ratio = X_raw_18[:, IDX_FWD_LEN]  / (X_raw_18[:, IDX_BWD_LEN]  + 1.0)
eng_feats = eng_scaler_2017.transform(np.column_stack([pkt_ratio, len_ratio]))

# ── ROBUST FEATURE DROPPING ──
X_18_scaled_by_2017[:, 0] = 0.0
X_18_scaled_by_2017[:, 1] = 0.0
X_18_scaled_by_2017[:, 3] = 0.0

X_test = np.column_stack([X_18_scaled_by_2017, eng_feats]).astype(np.float32)
logger.info(f"Final test set shape: {X_test.shape}")

# ── 4. Evaluation ─────────────────────────────────────────────────────────────
results_metrics = []
eval_models = {}

def evaluate_model(name: str, model_obj, X_eval: np.ndarray, y_eval: np.ndarray,
                   is_pytorch: bool = False, device='cpu') -> dict:
    logger.info(f"Evaluating {name} on CIC2018 (cross-dataset)...")
    t0 = time.time()

    if is_pytorch:
        model_obj.eval()
        Xt = torch.tensor(X_eval, dtype=torch.float32).to(device)
        with torch.no_grad():
            out = model_obj(Xt)
            probs  = torch.softmax(out, dim=1).cpu().numpy()
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

    # Convert to Binary: 0 = Benign, 1 = Attack (classes 1, 2, 3)
    y_eval_bin = (y_eval != 0).astype(int)
    y_pred_bin = (y_pred != 0).astype(int)
    
    # Probability of Attack is 1.0 - Probability(Benign)
    probs_bin = 1.0 - probs[:, 0]

    inf_time = time.time() - t0

    acc      = accuracy_score(y_eval_bin, y_pred_bin)
    prec     = precision_score(y_eval_bin, y_pred_bin, zero_division=0)
    rec      = recall_score(y_eval_bin, y_pred_bin, zero_division=0)
    macro_f1 = f1_score(y_eval_bin, y_pred_bin, zero_division=0) # Binary F1
    wt_f1    = f1_score(y_eval_bin, y_pred_bin, average='weighted', zero_division=0)
    bal_acc  = balanced_accuracy_score(y_eval_bin, y_pred_bin)
    mcc      = matthews_corrcoef(y_eval_bin, y_pred_bin)

    try:
        roc_auc = roc_auc_score(y_eval_bin, probs_bin)
    except Exception as e:
        logger.warning(f"ROC-AUC failed for {name}: {e}")
        roc_auc = np.nan

    try:
        pv, rv, _ = precision_recall_curve(y_eval_bin, probs_bin)
        pr_auc = auc(rv, pv)
    except Exception as e:
        logger.warning(f"PR-AUC failed for {name}: {e}")
        pr_auc = np.nan

    logger.info(f"  {name} (Binary): Acc={acc:.4f} | F1={macro_f1:.4f} "
                f"| ROC-AUC={roc_auc:.4f} | Inf={inf_time:.3f}s")

    # Confusion matrix
    cm = confusion_matrix(y_eval_bin, y_pred_bin)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
    plt.title(f"Binary Cross-Dataset Confusion Matrix — {name}\n(Train: CIC2017 → Test: CIC2018)")
    plt.ylabel('Actual (CIC2018)'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(CROSS_FIG_DIR,
                             f"binary_confusion_matrix_{name.lower().replace(' ','_')}.png"), dpi=300)
    plt.close()

    return {"Accuracy": acc, "Precision": prec, "Recall": rec,
            "Macro F1": macro_f1, "Weighted F1": wt_f1,
            "Balanced Accuracy": bal_acc, "MCC": mcc,
            "ROC-AUC": roc_auc, "PR-AUC": pr_auc,
            "Inference Time (s)": inf_time,
            "probs": probs_bin, "preds": y_pred_bin}

# ── 5. Load & Run Models ──────────────────────────────────────────────────────
models_to_eval = [
    ("Logistic Regression", "logistic_regression_robust.pkl", False),
    ("Extra Trees",         "extra_trees_robust.pkl",         False),
    ("LightGBM",            "lightgbm_robust.pkl",            False),
]

for name, fname, is_pt in models_to_eval:
    mp = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(mp):
        logger.error(f"Model not found: {mp} — skipping.")
        continue

    if is_pt:
        dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = MLPClassifierNet(input_dim=X_test.shape[1], num_classes=NUM_CLASSES).to(dev)
        model.load_state_dict(torch.load(mp, map_location=dev, weights_only=True))
        ev = evaluate_model(name, model, X_test, y_val_18, is_pytorch=True, device=dev)
        eval_models[name] = (model, ev)
    else:
        model = joblib.load(mp)
        if name == "LightGBM":
            model.n_jobs = 1
        ev = evaluate_model(name, model, X_test, y_val_18)
        eval_models[name] = (model, ev)

    results_metrics.append({"Model": name,
                             **{k: v for k, v in ev.items() if k not in ["probs","preds"]}})

# ── 6. Save Tables ────────────────────────────────────────────────────────────
df_cross = pd.DataFrame(results_metrics)
df_cross.to_csv(os.path.join(TABLE_DIR, "robust_binary_cross_2018_results.csv"), index=False)
logger.info("Saved: robust_binary_cross_2018_results.csv")

reports = []
for mname, (_, ev) in eval_models.items():
    pred_lbl = np.array(["Benign", "Attack"])[ev["preds"]]
    true_lbl = np.array(["Benign", "Attack"])[(y_val_18 != 0).astype(int)]
    rpt = classification_report(true_lbl, pred_lbl, output_dict=True, zero_division=0)
    for cls, m in rpt.items():
        if isinstance(m, dict):
            reports.append({"Model": mname, "Class": cls,
                            "Precision": m["precision"], "Recall": m["recall"],
                            "F1-Score": m["f1-score"], "Support": m["support"]})
pd.DataFrame(reports).to_csv(os.path.join(TABLE_DIR, "robust_binary_cross_2018_classification_reports.csv"), index=False)
logger.info("Saved: robust_binary_cross_2018_classification_reports.csv")

# ── 7. Visualizations ────────────────────────────────────────────────────────
logger.info("Generating binary cross-dataset ROC and PR curves...")

# ROC Curves
plt.figure(figsize=(8, 6))
y_val_18_bin = (y_val_18 != 0).astype(int)
for mname, (_, ev) in eval_models.items():
    probs_bin = ev["probs"]
    try:
        fpr, tpr, _ = roc_curve(y_val_18_bin, probs_bin)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{mname} (AUC={roc_auc:.3f})", lw=2)
    except Exception as e:
        logger.warning(f"Could not plot ROC for {mname}: {e}")
plt.plot([0,1],[0,1],'k--',lw=1.5)
plt.xlim([0,1]); plt.ylim([0,1.05])
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Binary Cross-Dataset ROC Curves\n(Train: CIC2017 → Test: CIC2018)')
plt.legend(loc="lower right"); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(CROSS_FIG_DIR, "binary_roc_curve.png"), dpi=300)
plt.close()

# PR Curves
plt.figure(figsize=(8, 6))
for mname, (_, ev) in eval_models.items():
    probs_bin = ev["probs"]
    try:
        pv, rv, _ = precision_recall_curve(y_val_18_bin, probs_bin)
        pr_a = auc(rv, pv)
        plt.plot(rv, pv, label=f"{mname} (PR-AUC={pr_a:.3f})", lw=2)
    except Exception as e:
        logger.warning(f"Could not plot PR for {mname}: {e}")
plt.xlim([0,1]); plt.ylim([0,1.05])
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.title('Binary Cross-Dataset PR Curves\n(Train: CIC2017 → Test: CIC2018)')
plt.legend(loc="lower left"); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(CROSS_FIG_DIR, "binary_pr_curve.png"), dpi=300)
plt.close()

logger.info("=" * 70)
logger.info("Binary Cross-Dataset Evaluation (CIC2017 → CIC2018) COMPLETE.")
logger.info("=" * 70)
