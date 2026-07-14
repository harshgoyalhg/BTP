"""
train_cic2017.py — Stage 6: CICIDS-2017 Training Pipeline
==========================================================
Mirrors train_cic2018.py (Stage 5) exactly in methodology:
  • Chunked CSV loading with benign downsampling (2%)
  • Stratified 500k sample
  • 80/20 train/val split
  • Feature engineering (fwd/bwd packet ratio + length ratio)
  • 8 models with IDENTICAL hyperparameters to Stage 5
  • Full metrics: Acc, Prec, Rec, F1, Bal-Acc, MCC, ROC-AUC, PR-AUC
  • Confusion matrices, ROC curves, PR curves, feature importance
  • All artifacts saved to stage6_cic2017_training/ (stage5 untouched)

Key differences vs. Stage 5:
  • 2017 columns already match SCALER_FEATURES — no renaming map needed
  • Duplicate 'Fwd Header Length' in 2017 CSVs auto-renamed to '.1' by pandas
  • PortScan exists natively — no injection from another dataset required
  • Fits its own scaler, label_encoder, and engineered_scaler
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import glob
import time
import joblib
import psutil
import threading
import logging

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix, classification_report,
    roc_curve
)
import xgboost as xgb
import lightgbm as lgb

# ── import shared config ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    BASE_DIR, ARTIFACT_DIR, MODEL_DIR, TABLE_DIR, LOG_DIR, REPORT_DIR, FIG_DIR,
    DATASET_2017, SCALER_FEATURES, LABEL_MAPPING_2017, COLS_TO_DROP,
    HP, CLASSES, NUM_CLASSES, INPUT_DIM_ENG,
    IDX_FWD_PKTS, IDX_BWD_PKTS, IDX_FWD_LEN, IDX_BWD_LEN,
    STRATIFIED_SAMPLE_SIZE, BENIGN_DOWNSAMPLE_FRAC, CHUNK_SIZE,
    TRAIN_TEST_SPLIT_RATIO, RANDOM_STATE
)

# ── 1. Directory Setup & Logging ──────────────────────────────────────────────
for d in [ARTIFACT_DIR, MODEL_DIR, TABLE_DIR, LOG_DIR, REPORT_DIR,
          FIG_DIR, os.path.join(FIG_DIR, "cross_2018"),
          os.path.join(FIG_DIR, "ensemble"), os.path.join(FIG_DIR, "lycos")]:
    os.makedirs(d, exist_ok=True)

log_format = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("stage6_training")
logger.info("=" * 70)
logger.info("Stage 6: CICIDS-2017 Training Pipeline")
logger.info("=" * 70)

# ── 2. Memory Tracker ─────────────────────────────────────────────────────────
class MemoryTracker:
    def __init__(self):
        self.max_memory = 0.0
        self.running = False
        self.thread = None

    def _track(self):
        proc = psutil.Process(os.getpid())
        while self.running:
            try:
                mem = proc.memory_info().rss / (1024 * 1024)
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


# ── 3. Chunked Loader ─────────────────────────────────────────────────────────
def load_and_preprocess_cic2017(dataset_dir: str) -> pd.DataFrame:
    """
    Load all 8 CICIDS-2017 CSV files in memory-safe 200k-row chunks.

    Key handling:
    - Drop metadata columns (Flow ID, IP, Timestamp)
    - The 2017 CSVs have 'Fwd Header Length' appearing twice (cols 40 & 59).
      After dropping metadata cols, pandas auto-renames the duplicate to
      'Fwd Header Length.1' which is exactly the name in SCALER_FEATURES.
    - Map all raw labels to 4-class schema
    - Downsample benign traffic to 2% per chunk to prevent RAM issues
    """
    csv_files = sorted(glob.glob(os.path.join(dataset_dir, "*.csv")))
    logger.info(f"Found {len(csv_files)} CSV files in dataset directory.")

    attack_dfs = []
    benign_dfs = []

    for f in csv_files:
        filename = os.path.basename(f)
        logger.info(f"Processing: {filename}")
        chunk_idx = 0

        for chunk in pd.read_csv(f, chunksize=CHUNK_SIZE, low_memory=False,
                                  encoding='utf-8', encoding_errors='replace'):
            chunk_idx += 1
            chunk.columns = chunk.columns.str.strip()

            # Remove duplicate header rows that appear mid-file
            if 'Label' not in chunk.columns:
                continue
            chunk = chunk[chunk['Label'] != 'Label']

            # Drop metadata columns not used as features
            drop_cols = [c for c in COLS_TO_DROP if c in chunk.columns]
            chunk.drop(columns=drop_cols, inplace=True)

            # Handle the duplicate 'Fwd Header Length' column.
            # pandas renames the second occurrence automatically to
            # 'Fwd Header Length.1' when reading, but after stripping
            # column names the renaming may be intact. Verify:
            fhl_count = sum(1 for c in chunk.columns if c == 'Fwd Header Length')
            if fhl_count == 2:
                # Both named the same — manually rename second occurrence
                cols = list(chunk.columns)
                first_idx = cols.index('Fwd Header Length')
                second_idx = len(cols) - 1 - cols[::-1].index('Fwd Header Length')
                if first_idx != second_idx:
                    cols[second_idx] = 'Fwd Header Length.1'
                    chunk.columns = cols

            # Ensure 'Fwd Header Length.1' exists (clone if missing)
            if 'Fwd Header Length.1' not in chunk.columns and 'Fwd Header Length' in chunk.columns:
                chunk['Fwd Header Length.1'] = chunk['Fwd Header Length']

            # Keep only the 80 SCALER_FEATURES + Label
            available = [c for c in SCALER_FEATURES if c in chunk.columns]
            chunk = chunk[available + ['Label']]

            # Fill any entirely absent feature columns with 0
            for col in SCALER_FEATURES:
                if col not in chunk.columns:
                    chunk[col] = 0.0
            chunk = chunk[SCALER_FEATURES + ['Label']]

            # Convert to float32, replace inf → NaN
            feat_df = chunk[SCALER_FEATURES].apply(pd.to_numeric, errors='coerce').astype(np.float32)
            feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Convert Label to string first — some rows from encoding issues may
            # have numeric values in Label column
            label_series = chunk['Label'].astype(str).str.strip()
            cleaned = pd.concat([feat_df, label_series], axis=1)
            cleaned.dropna(inplace=True)
            if cleaned.empty:
                continue

            # Map labels to 4-class schema
            cleaned['MappedLabel'] = cleaned['Label'].map(LABEL_MAPPING_2017)
            cleaned.dropna(subset=['MappedLabel'], inplace=True)
            cleaned.drop(columns=['Label'], inplace=True)
            cleaned.rename(columns={'MappedLabel': 'Label'}, inplace=True)

            # Split benign vs attack
            benign  = cleaned[cleaned['Label'] == 'BENIGN']
            attacks = cleaned[cleaned['Label'] != 'BENIGN']

            if not attacks.empty:
                attack_dfs.append(attacks)
            if not benign.empty:
                # Downsample benign to 2% per chunk (same as Stage 5)
                if len(benign) > 1000:
                    benign = benign.sample(frac=BENIGN_DOWNSAMPLE_FRAC, random_state=RANDOM_STATE)
                benign_dfs.append(benign)

        logger.info(f"  Finished {filename} — {chunk_idx} chunks processed.")

    if not attack_dfs:
        raise RuntimeError("No attack samples loaded. Check dataset path.")

    all_attacks = pd.concat(attack_dfs, ignore_index=True)
    all_benign  = pd.concat(benign_dfs,  ignore_index=True)
    logger.info(f"Attack rows:  {len(all_attacks):,}")
    logger.info(f"Benign rows (downsampled): {len(all_benign):,}")

    combined = pd.concat([all_attacks, all_benign], ignore_index=True)
    logger.info(f"Combined shape: {combined.shape}")
    return combined


# ── 4. Load or Preprocess ─────────────────────────────────────────────────────
NPZ_CACHE = os.path.join(ARTIFACT_DIR, "cicids2017_processed.npz")

if os.path.exists(NPZ_CACHE):
    logger.info("Loading preprocessed dataset from cache (stage6 artifacts)...")
    cached = np.load(NPZ_CACHE, allow_pickle=True)
    X_train_scaled = cached['X_train']
    X_val_scaled   = cached['X_val']
    y_train        = cached['y_train']
    y_val          = cached['y_val']

    le     = joblib.load(os.path.join(ARTIFACT_DIR, "label_encoder.pkl"))
    scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    logger.info(f"Train shape: {X_train_scaled.shape} | Val shape: {X_val_scaled.shape}")
else:
    logger.info("Step 1: Reading and pre-processing CICIDS-2017 dataset...")
    raw_data = load_and_preprocess_cic2017(DATASET_2017)

    # Stratified sample of up to 500k rows (same as Stage 5)
    target_size = min(STRATIFIED_SAMPLE_SIZE, len(raw_data))
    logger.info(f"Step 2: Stratified sample of {target_size:,} rows...")
    _, sampled_df = train_test_split(
        raw_data,
        test_size=target_size,
        random_state=RANDOM_STATE,
        stratify=raw_data['Label']
    )
    logger.info(f"Class distribution:\n{sampled_df['Label'].value_counts()}")

    X = sampled_df[SCALER_FEATURES]
    y_labels = sampled_df['Label']

    # 80/20 stratified split
    X_train, X_val, y_train_lbl, y_val_lbl = train_test_split(
        X, y_labels,
        test_size=TRAIN_TEST_SPLIT_RATIO,
        random_state=RANDOM_STATE,
        stratify=y_labels
    )

    # Fit label encoder on all 4 classes
    le = LabelEncoder()
    le.fit(CLASSES)
    y_train = le.transform(y_train_lbl).astype(np.int64)
    y_val   = le.transform(y_val_lbl).astype(np.int64)

    # Fit scaler on training set only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled   = scaler.transform(X_val).astype(np.float32)

    # Save transformers
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    joblib.dump(le,     os.path.join(ARTIFACT_DIR, "label_encoder.pkl"))

    # Cache processed data
    logger.info("Saving preprocessed dataset to cache...")
    np.savez_compressed(NPZ_CACHE,
                        X_train=X_train_scaled, X_val=X_val_scaled,
                        y_train=y_train,        y_val=y_val)

logger.info(f"Train: {X_train_scaled.shape} | labels: {np.bincount(y_train)}")
logger.info(f"Val:   {X_val_scaled.shape}   | labels: {np.bincount(y_val)}")
logger.info(f"Classes: {le.classes_}")


# ── 5. Feature Engineering (identical to Stage 5) ────────────────────────────
logger.info("Step 3: Applying feature engineering (Fwd/Bwd packet & length ratios)...")

engineered_scaler = None  # will be set by add_engineered_features on first call

def add_engineered_features(X_scaled: np.ndarray, is_train: bool = True) -> np.ndarray:
    """
    Inverse-transform → compute 2 ratio features → standardize → append.
    Returns array of shape (N, 82).

    Ratios:
      pkt_ratio = Total Fwd Packets  / (Total Bwd Packets + 1)
      len_ratio = Total Fwd Pkt Len  / (Total Bwd Pkt Len + 1)

    This is identical to Stage 5 feature engineering.
    """
    global engineered_scaler
    X_raw = scaler.inverse_transform(X_scaled)
    pkt_ratio = X_raw[:, IDX_FWD_PKTS] / (X_raw[:, IDX_BWD_PKTS] + 1.0)
    len_ratio = X_raw[:, IDX_FWD_LEN]  / (X_raw[:, IDX_BWD_LEN]  + 1.0)
    new_feats = np.column_stack([pkt_ratio, len_ratio])

    if is_train:
        engineered_scaler = StandardScaler()
        new_feats_scaled = engineered_scaler.fit_transform(new_feats)
        joblib.dump(engineered_scaler, os.path.join(ARTIFACT_DIR, "engineered_scaler_robust.pkl"))
        logger.info("Fitted and saved engineered_scaler_robust.pkl")
    else:
        new_feats_scaled = engineered_scaler.transform(new_feats)

    # ── ROBUST FEATURE DROPPING ──
    # 0: Source Port, 1: Destination Port, 3: Flow Duration
    X_scaled[:, 0] = 0.0
    X_scaled[:, 1] = 0.0
    X_scaled[:, 3] = 0.0

    return np.column_stack([X_scaled, new_feats_scaled]).astype(np.float32)

X_train_scaled = add_engineered_features(X_train_scaled, is_train=True)
X_val_scaled   = add_engineered_features(X_val_scaled,   is_train=False)
logger.info(f"After FE — Train: {X_train_scaled.shape} | Val: {X_val_scaled.shape}")


# ── 6. MLP Architecture (identical to Stage 5) ───────────────────────────────
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


# ── 7. Evaluation Function ────────────────────────────────────────────────────
results_metrics = []
trained_models  = {}

def evaluate_model(name: str, model_obj, X_test: np.ndarray, y_test: np.ndarray,
                   is_pytorch: bool = False, device: str = 'cpu') -> dict:
    logger.info(f"Evaluating {name}...")
    t0 = time.time()

    if is_pytorch:
        model_obj.eval()
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            out   = model_obj(X_t)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            y_pred = np.argmax(probs, axis=1)
    else:
        y_pred = model_obj.predict(X_test)
        if hasattr(model_obj, "predict_proba"):
            probs = model_obj.predict_proba(X_test)
        else:
            # LinearSVC fallback — softmax over decision function
            dec = model_obj.decision_function(X_test)
            if len(dec.shape) == 1:
                p = 1 / (1 + np.exp(-dec))
                probs = np.column_stack([1 - p, p])
            else:
                exp_d = np.exp(dec - np.max(dec, axis=1, keepdims=True))
                probs = exp_d / exp_d.sum(axis=1, keepdims=True)

    inf_time = time.time() - t0

    acc       = accuracy_score(y_test, y_pred)
    prec      = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec       = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    macro_f1  = f1_score(y_test, y_pred, average='macro', zero_division=0)
    wt_f1     = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    bal_acc   = balanced_accuracy_score(y_test, y_pred)
    mcc       = matthews_corrcoef(y_test, y_pred)

    try:
        roc_auc = roc_auc_score(y_test, probs, multi_class='ovr', average='macro')
    except Exception as e:
        logger.warning(f"ROC-AUC not computed for {name}: {e}")
        roc_auc = np.nan

    try:
        pr_list = []
        for i in range(NUM_CLASSES):
            y_b = (y_test == i).astype(int)
            p_v, r_v, _ = precision_recall_curve(y_b, probs[:, i])
            pr_list.append(auc(r_v, p_v))
        pr_auc = np.mean(pr_list)
    except Exception as e:
        logger.warning(f"PR-AUC not computed for {name}: {e}")
        pr_auc = np.nan

    logger.info(f"  {name}: Acc={acc:.4f} | Wt-F1={wt_f1:.4f} | Macro-F1={macro_f1:.4f} "
                f"| MCC={mcc:.4f} | Inf={inf_time:.3f}s")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"confusion_matrix_{name.lower().replace(' ', '_')}.png"), dpi=300)
    plt.close()

    return {
        "Accuracy": acc, "Precision": prec, "Recall": rec,
        "Macro F1": macro_f1, "Weighted F1": wt_f1,
        "Balanced Accuracy": bal_acc, "MCC": mcc,
        "ROC-AUC": roc_auc, "PR-AUC": pr_auc,
        "Inference Time": inf_time,
        "CM": cm, "probs": probs, "preds": y_pred
    }


# ── 8. Training ───────────────────────────────────────────────────────────────

# ── A. Logistic Regression ────────────────────────────────────────────────────
logger.info("\n=== Training Logistic Regression ===")
lr_cfg = HP['logistic_regression']
lr_model = LogisticRegression(**lr_cfg)
mt = MemoryTracker(); mt.start(); t0 = time.time()
lr_model.fit(X_train_scaled, y_train)
train_time = time.time() - t0; peak_ram = mt.stop()
logger.info(f"LR done in {train_time:.2f}s | Peak RAM: {peak_ram:.2f} MB")
mp = os.path.join(MODEL_DIR, "logistic_regression_robust.pkl")
joblib.dump(lr_model, mp)
evals = evaluate_model("Logistic Regression", lr_model, X_val_scaled, y_val)
results_metrics.append({"Model": "Logistic Regression", "Train Time (s)": train_time,
    "Inference Time (s)": evals["Inference Time"], "Peak RAM (MB)": peak_ram,
    "Model Size (MB)": os.path.getsize(mp)/(1024*1024),
    **{k: v for k, v in evals.items() if k not in ["CM","probs","preds","Inference Time"]}})
trained_models["Logistic Regression"] = (lr_model, evals)


# ── C. Extra Trees ────────────────────────────────────────────────────────────
logger.info("\n=== Training Extra Trees ===")
et_model = ExtraTreesClassifier(**HP['extra_trees'])
mt = MemoryTracker(); mt.start(); t0 = time.time()
et_model.fit(X_train_scaled, y_train)
train_time = time.time() - t0; peak_ram = mt.stop()
logger.info(f"ET done in {train_time:.2f}s | Peak RAM: {peak_ram:.2f} MB")
mp = os.path.join(MODEL_DIR, "extra_trees_robust.pkl")
joblib.dump(et_model, mp)
evals = evaluate_model("Extra Trees", et_model, X_val_scaled, y_val)
results_metrics.append({"Model": "Extra Trees", "Train Time (s)": train_time,
    "Inference Time (s)": evals["Inference Time"], "Peak RAM (MB)": peak_ram,
    "Model Size (MB)": os.path.getsize(mp)/(1024*1024),
    **{k: v for k, v in evals.items() if k not in ["CM","probs","preds","Inference Time"]}})
trained_models["Extra Trees"] = (et_model, evals)


# ── E. LightGBM ───────────────────────────────────────────────────────────────
logger.info("\n=== Training LightGBM ===")
lgb_model = lgb.LGBMClassifier(**HP['lightgbm'])
mt = MemoryTracker(); mt.start(); t0 = time.time()
lgb_model.fit(X_train_scaled, y_train)
train_time = time.time() - t0; peak_ram = mt.stop()
logger.info(f"LGBM done in {train_time:.2f}s | Peak RAM: {peak_ram:.2f} MB")
mp = os.path.join(MODEL_DIR, "lightgbm_robust.pkl")
joblib.dump(lgb_model, mp)
evals = evaluate_model("LightGBM", lgb_model, X_val_scaled, y_val)
results_metrics.append({"Model": "LightGBM", "Train Time (s)": train_time,
    "Inference Time (s)": evals["Inference Time"], "Peak RAM (MB)": peak_ram,
    "Model Size (MB)": os.path.getsize(mp)/(1024*1024),
    **{k: v for k, v in evals.items() if k not in ["CM","probs","preds","Inference Time"]}})
trained_models["LightGBM"] = (lgb_model, evals)




# ── 9. Generate Tables ────────────────────────────────────────────────────────
logger.info("\n=== Compiling Result Tables ===")
df_metrics = pd.DataFrame(results_metrics)
df_metrics.to_csv(os.path.join(TABLE_DIR, "training_results.csv"),  index=False)
df_metrics.to_csv(os.path.join(TABLE_DIR, "model_comparison.csv"),  index=False)

df_metrics[["Model","Train Time (s)","Inference Time (s)","Peak RAM (MB)","Model Size (MB)"]]\
    .to_csv(os.path.join(TABLE_DIR, "resource_usage.csv"), index=False)

# Per-class classification reports
reports = []
for mname, (_, ev) in trained_models.items():
    pred_lbl = le.classes_[ev["preds"]]
    true_lbl = le.classes_[y_val]
    rpt = classification_report(true_lbl, pred_lbl, output_dict=True, zero_division=0)
    for cls, m in rpt.items():
        if isinstance(m, dict):
            reports.append({"Model": mname, "Class": cls,
                            "Precision": m["precision"], "Recall": m["recall"],
                            "F1-Score": m["f1-score"], "Support": m["support"]})
pd.DataFrame(reports).to_csv(os.path.join(TABLE_DIR, "classification_reports.csv"), index=False)

# Feature importance (tree-based models)
fi = {"Feature": SCALER_FEATURES + ['Fwd/Bwd Packet Ratio', 'Fwd/Bwd Length Ratio']}
for mname, (mo, _) in trained_models.items():
    if hasattr(mo, "feature_importances_"):
        fi[mname] = mo.feature_importances_
df_fi = pd.DataFrame(fi)
df_fi.to_csv(os.path.join(TABLE_DIR, "feature_importance.csv"), index=False)
logger.info("Tables saved.")


# ── 10. Visualizations ────────────────────────────────────────────────────────
logger.info("Generating publication-quality figures...")

# A. Macro-Average ROC Curves
plt.figure(figsize=(8, 6))
for mname, (_, ev) in trained_models.items():
    probs = ev["probs"]
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    for i in range(NUM_CLASSES):
        yb = (y_val == i).astype(int)
        if yb.sum() == 0:
            continue
        fpr_, tpr_, _ = roc_curve(yb, probs[:, i])
        tprs.append(np.interp(mean_fpr, fpr_, tpr_))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[0] = 0.0
    macro_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, label=f"{mname} (AUC={macro_auc:.3f})", lw=2)
plt.plot([0,1],[0,1],'k--',lw=1.5)
plt.xlim([0,1]); plt.ylim([0,1.05])
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Macro-Average ROC Curves (CICIDS2017 Validation)')
plt.legend(loc="lower right"); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "roc_curve.png"), dpi=300)
plt.close()

# B. Macro-Average PR Curves
plt.figure(figsize=(8, 6))
for mname, (_, ev) in trained_models.items():
    probs = ev["probs"]
    mean_rec = np.linspace(0, 1, 100)
    precs = []
    for i in range(NUM_CLASSES):
        yb = (y_val == i).astype(int)
        if yb.sum() == 0:
            continue
        pv, rv, _ = precision_recall_curve(yb, probs[:, i])
        precs.append(np.interp(mean_rec, rv[::-1], pv[::-1]))
    mean_prec = np.mean(precs, axis=0)
    pr_a = auc(mean_rec, mean_prec)
    plt.plot(mean_rec, mean_prec, label=f"{mname} (PR-AUC={pr_a:.3f})", lw=2)
plt.xlim([0,1]); plt.ylim([0,1.05])
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.title('Macro-Average PR Curves (CICIDS2017 Validation)')
plt.legend(loc="lower left"); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "pr_curve.png"), dpi=300)
plt.close()

# C. Top-15 Feature Importance (XGBoost)
top_n = 15
if "XGBoost" in df_fi.columns:
    df_fi_sorted = df_fi.sort_values("XGBoost", ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_fi_sorted, y="Feature", x="XGBoost", palette="viridis")
    plt.title(f"Top {top_n} Feature Importances — XGBoost (CICIDS2017)")
    plt.xlabel("Importance Score"); plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "feature_importance.png"), dpi=300)
    plt.close()

# D. Resource Comparison
metrics_to_plot = ["Train Time (s)", "Inference Time (s)", "Peak RAM (MB)", "Model Size (MB)"]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for idx, col in enumerate(metrics_to_plot):
    sns.barplot(data=df_metrics, x="Model", y=col, ax=axes[idx], palette="muted")
    axes[idx].set_title(col); axes[idx].set_xlabel("")
    axes[idx].grid(True, alpha=0.3)
    for lbl in axes[idx].get_xticklabels():
        lbl.set_rotation(15)
plt.suptitle("Resource Profiling Comparison — CICIDS2017 (Stage 6)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "metrics_comparison.png"), dpi=300)
plt.close()

logger.info("=" * 70)
logger.info("Stage 6 training pipeline COMPLETE. All artifacts saved.")
logger.info("=" * 70)
