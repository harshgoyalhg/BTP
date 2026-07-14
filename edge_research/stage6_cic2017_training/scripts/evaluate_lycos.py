"""
evaluate_lycos.py — Stage 6: LycoS External Dataset Evaluation
================================================================
Evaluates all 8 Stage-6 models (trained on CICIDS-2017) on the
LycoS-Unicas-IDS2018 external dataset.

Since LycoS was pre-processed and cached by Stage 5 using the 2018 scaler,
this script:
  1. Loads stage5's lycos_processed.npz (scaled by 2018 scaler)
  2. Inverse-transforms with 2018 scaler → raw features
  3. Re-applies stage6's 2017 scaler
  4. Applies stage6's engineered_scaler (FE)
  5. Evaluates all 8 stage6 models

Note: LycoS contains only BENIGN, Bot, DDoS — PortScan is absent.
      This is handled gracefully in all metrics computations.
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
    precision_recall_curve, auc, confusion_matrix, classification_report, roc_curve
)

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    BASE_DIR, ARTIFACT_DIR, MODEL_DIR, TABLE_DIR, LOG_DIR, FIG_DIR,
    STAGE5_ARTIFACT_DIR, SCALER_FEATURES, CLASSES, NUM_CLASSES,
    IDX_FWD_PKTS, IDX_BWD_PKTS, IDX_FWD_LEN, IDX_BWD_LEN
)

# ── Setup ─────────────────────────────────────────────────────────────────────
LYCOS_FIG_DIR = os.path.join(FIG_DIR, "lycos")
for d in [LYCOS_FIG_DIR, TABLE_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "lycos_evaluation.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("stage6_lycos")
logger.info("=" * 70)
logger.info("Stage 6 — LycoS External Dataset Evaluation")
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

# ── 1. Load Transformers ──────────────────────────────────────────────────────
logger.info("Loading transformers...")
scaler_2018     = joblib.load(os.path.join(STAGE5_ARTIFACT_DIR, "scaler.pkl"))
scaler_2017     = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
eng_scaler_2017 = joblib.load(os.path.join(ARTIFACT_DIR, "engineered_scaler.pkl"))
le              = joblib.load(os.path.join(ARTIFACT_DIR, "label_encoder.pkl"))
logger.info(f"Classes: {le.classes_}")

# ── 2. Load LycoS (from stage5 cache) ────────────────────────────────────────
logger.info("Loading LycoS from stage5 artifacts cache...")
data_lycos = np.load(os.path.join(STAGE5_ARTIFACT_DIR, "lycos_processed.npz"))
X_lycos_s5 = data_lycos['X_test'].astype(np.float32)  # scaled by 2018 scaler
y_test      = data_lycos['y_test'].astype(np.int64)
logger.info(f"LycoS shape (stage5 cache): {X_lycos_s5.shape}, labels: {np.bincount(y_test)}")

# ── 3. Re-scale LycoS: 2018 space → 2017 space → FE → 82-dim ────────────────
logger.info("Inverse-transforming with 2018 scaler to recover raw features...")
X_lycos_raw = scaler_2018.inverse_transform(X_lycos_s5)

logger.info("Applying Stage 6 (2017) scaler...")
X_lycos_scaled = scaler_2017.transform(X_lycos_raw).astype(np.float32)

logger.info("Applying feature engineering (2017 engineered_scaler)...")
pkt_ratio = X_lycos_raw[:, IDX_FWD_PKTS] / (X_lycos_raw[:, IDX_BWD_PKTS] + 1.0)
len_ratio = X_lycos_raw[:, IDX_FWD_LEN]  / (X_lycos_raw[:, IDX_BWD_LEN]  + 1.0)
eng_feats = eng_scaler_2017.transform(np.column_stack([pkt_ratio, len_ratio]))
X_test = np.column_stack([X_lycos_scaled, eng_feats]).astype(np.float32)
logger.info(f"Final LycoS test set shape: {X_test.shape}")

# ── 4. Evaluation ─────────────────────────────────────────────────────────────
results = []
eval_models = {}

# LycoS active classes: only BENIGN(0), Bot(1), DDoS(2) — PortScan(3) absent
LYCOS_ACTIVE = [i for i in range(NUM_CLASSES) if np.sum(y_test == i) > 0]
logger.info(f"Active classes in LycoS: {[CLASSES[i] for i in LYCOS_ACTIVE]}")

def evaluate_model(name: str, model_obj, X_eval: np.ndarray, y_eval: np.ndarray,
                   is_pytorch: bool = False, device='cpu') -> dict:
    logger.info(f"Evaluating {name} on LycoS...")
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

    inf_time = time.time() - t0

    acc      = accuracy_score(y_eval, y_pred)
    prec     = precision_score(y_eval, y_pred, average='weighted', zero_division=0)
    rec      = recall_score(y_eval, y_pred, average='weighted', zero_division=0)
    macro_f1 = f1_score(y_eval, y_pred, average='macro', zero_division=0)
    wt_f1    = f1_score(y_eval, y_pred, average='weighted', zero_division=0)
    bal_acc  = balanced_accuracy_score(y_eval, y_pred)
    mcc      = matthews_corrcoef(y_eval, y_pred)

    try:
        roc_auc = roc_auc_score(y_eval, probs, multi_class='ovr', average='macro')
    except Exception as e:
        logger.warning(f"ROC-AUC failed for {name}: {e}"); roc_auc = np.nan

    try:
        pr_list = []
        for i in LYCOS_ACTIVE:
            yb = (y_eval == i).astype(int)
            if yb.sum() == 0: continue
            pv, rv, _ = precision_recall_curve(yb, probs[:, i])
            pr_list.append(auc(rv, pv))
        pr_auc = np.mean(pr_list) if pr_list else np.nan
    except Exception as e:
        logger.warning(f"PR-AUC failed for {name}: {e}"); pr_auc = np.nan

    logger.info(f"  {name}: Acc={acc:.4f} | Wt-F1={wt_f1:.4f} | Macro-F1={macro_f1:.4f} "
                f"| MCC={mcc:.4f} | Inf={inf_time:.3f}s")

    # Confusion matrix
    cm = confusion_matrix(y_eval, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"LycoS Confusion Matrix — {name}\n(Train: CIC2017 → Test: LycoS)")
    plt.ylabel('Actual (LycoS)'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(LYCOS_FIG_DIR,
                             f"confusion_matrix_{name.lower().replace(' ','_')}.png"), dpi=300)
    plt.close()

    return {"Accuracy": acc, "Precision": prec, "Recall": rec,
            "Macro F1": macro_f1, "Weighted F1": wt_f1,
            "Balanced Accuracy": bal_acc, "MCC": mcc,
            "ROC-AUC": roc_auc, "PR-AUC": pr_auc,
            "Inference Time (s)": inf_time,
            "probs": probs, "preds": y_pred}

# ── 5. Load & Run Models ──────────────────────────────────────────────────────
models_to_eval = [
    ("Logistic Regression", "logistic_regression.pkl", False),
    ("Random Forest",       "random_forest.pkl",       False),
    ("Extra Trees",         "extra_trees.pkl",         False),
    ("XGBoost",             "xgboost.pkl",             False),
    ("LightGBM",            "lightgbm.pkl",            False),
    ("MLP (PyTorch)",       "mlp.pt",                  True),
    ("SVM (Linear)",        "svm_linear.pkl",          False),
    ("SVM (RBF Kernel)",    "svm_rbf.pkl",             False),
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
        ev = evaluate_model(name, model, X_test, y_test, is_pytorch=True, device=dev)
        eval_models[name] = (model, ev)
    else:
        model = joblib.load(mp)
        if name == "LightGBM":
            model.n_jobs = 1
        ev = evaluate_model(name, model, X_test, y_test)
        eval_models[name] = (model, ev)

    results.append({"Model": name,
                    **{k: v for k, v in ev.items() if k not in ["probs","preds"]}})

# ── 6. Save Tables ────────────────────────────────────────────────────────────
df_res = pd.DataFrame(results)
df_res.to_csv(os.path.join(TABLE_DIR, "lycos_results.csv"), index=False)
logger.info("Saved: lycos_results.csv")

reports = []
for mname, (_, ev) in eval_models.items():
    pred_lbl = np.array(CLASSES)[ev["preds"]]
    true_lbl = np.array(CLASSES)[y_test]
    rpt = classification_report(true_lbl, pred_lbl, output_dict=True, zero_division=0)
    for cls, m in rpt.items():
        if isinstance(m, dict):
            reports.append({"Model": mname, "Class": cls,
                            "Precision": m["precision"], "Recall": m["recall"],
                            "F1-Score": m["f1-score"], "Support": m["support"]})
pd.DataFrame(reports).to_csv(os.path.join(TABLE_DIR, "lycos_classification_reports.csv"), index=False)
logger.info("Saved: lycos_classification_reports.csv")

# ── 7. ROC Curves ────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
for mname, (_, ev) in eval_models.items():
    probs = ev["probs"]
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    for i in LYCOS_ACTIVE:
        yb = (y_test == i).astype(int)
        if yb.sum() == 0: continue
        fpr_, tpr_, _ = roc_curve(yb, probs[:, i])
        tprs.append(np.interp(mean_fpr, fpr_, tpr_))
    if not tprs: continue
    mean_tpr = np.mean(tprs, axis=0); mean_tpr[0] = 0.0
    macro_a = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, label=f"{mname} (AUC={macro_a:.3f})", lw=2)
plt.plot([0,1],[0,1],'k--',lw=1.5)
plt.xlim([0,1]); plt.ylim([0,1.05])
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curves — LycoS External\n(Train: CIC2017 → Test: LycoS)')
plt.legend(loc="lower right"); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(LYCOS_FIG_DIR, "roc_curve.png"), dpi=300); plt.close()

# ── 8. PR Curves ─────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
for mname, (_, ev) in eval_models.items():
    probs = ev["probs"]
    mean_rec = np.linspace(0, 1, 100)
    prec_list = []
    for i in LYCOS_ACTIVE:
        yb = (y_test == i).astype(int)
        if yb.sum() == 0: continue
        pv, rv, _ = precision_recall_curve(yb, probs[:, i])
        prec_list.append(np.interp(mean_rec, rv[::-1], pv[::-1]))
    if not prec_list: continue
    mean_p = np.mean(prec_list, axis=0)
    pa = auc(mean_rec, mean_p)
    plt.plot(mean_rec, mean_p, label=f"{mname} (PR-AUC={pa:.3f})", lw=2)
plt.xlim([0,1]); plt.ylim([0,1.05])
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.title('PR Curves — LycoS External\n(Train: CIC2017 → Test: LycoS)')
plt.legend(loc="lower left"); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(LYCOS_FIG_DIR, "pr_curve.png"), dpi=300); plt.close()

logger.info("=" * 70)
logger.info("Stage 6 LycoS Evaluation COMPLETE.")
logger.info("=" * 70)
