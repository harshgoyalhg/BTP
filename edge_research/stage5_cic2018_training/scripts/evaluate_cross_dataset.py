import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import joblib
import psutil
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix, classification_report
)

# 1. Setup Directories & Logging
BASE_DIR = "/Users/harshgoyal/Documents/BTP/BTP/edge_research/stage5_cic2018_training"
os.makedirs(os.path.join(BASE_DIR, "figures", "cross_dataset"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "tables"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

log_format = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "logs", "cross_validation.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("cross_dataset_eval")

logger.info("Initializing Cross-Dataset Validation (Train: CICIDS2018 -> Test: CICIDS2017)...")

# Classes ordered matching the Stage 2 label encoder
CLASSES = ['BENIGN', 'Bot', 'DDoS', 'PortScan']

# 2. PyTorch Network Architecture (matching stage5 training)
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

# 3. Load preprocessed CICIDS2017 test dataset
processed_2017_path = os.path.join(BASE_DIR, "artifacts", "cicids2017_processed.npz")
logger.info(f"Loading preprocessed CICIDS2017 dataset from {processed_2017_path}...")
data_2017 = np.load(processed_2017_path)
X_test = data_2017['X_test'].astype(np.float32)
y_test = data_2017['y_test'].astype(np.int64)
logger.info(f"Loaded test set shape: {X_test.shape}, labels distribution: {np.bincount(y_test)}")

# 3.5. Feature Engineering (Unscaling, computing ratios, scaling, and appending)
logger.info("Applying feature engineering (ratios on raw features) for test set...")
# Load scalers
processed_dir = os.path.join(BASE_DIR, "artifacts")
scaler = joblib.load(os.path.join(processed_dir, "scaler.pkl"))
engineered_scaler = joblib.load(os.path.join(processed_dir, "engineered_scaler.pkl"))

# Define SCALER_FEATURES matching stage5 training
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
logger.info(f"After Feature Engineering - test set shape: {X_test.shape}")


# 4. Evaluation Loop
results_metrics = []
trained_models_eval = {}

def evaluate_model(name, model_obj, X_eval, y_eval, is_pytorch=False, device='cpu'):
    logger.info(f"Evaluating model: {name} on CICIDS2017...")
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
        logger.warning(f"Could not compute ROC-AUC for {name}: {e}")
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
        logger.warning(f"Could not compute PR-AUC for {name}: {e}")
        pr_auc = np.nan
        
    logger.info(f"{name} Results: Acc={acc:.4f}, Weighted-F1={weighted_f1:.4f}, Macro-F1={macro_f1:.4f}, InfTime={inference_time:.3f}s")
    
    # Save Confusion Matrix Heatmap
    cm = confusion_matrix(y_eval, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"Cross-Dataset Confusion Matrix - {name}")
    plt.ylabel('Actual (CICIDS2017)')
    plt.xlabel('Predicted (by CICIDS2018 model)')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "figures", "cross_dataset", f"confusion_matrix_{name.lower().replace(' ', '_')}.png"), dpi=300)
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
models_to_evaluate = [
    ("Logistic Regression", "logistic_regression.pkl", False),
    ("Random Forest", "random_forest.pkl", False),
    ("Extra Trees", "extra_trees.pkl", False),
    ("XGBoost", "xgboost.pkl", False),
    ("LightGBM", "lightgbm.pkl", False),
    ("MLP (PyTorch)", "mlp.pt", True),
    ("SVM (Linear)", "svm_linear.pkl", False),
    ("SVM (RBF Kernel)", "svm_rbf.pkl", False),
]

for name, filename, is_pytorch in models_to_evaluate:
    model_path = os.path.join(BASE_DIR, "models", filename)
    if not os.path.exists(model_path):
        logger.error(f"Model file {model_path} does not exist. Skipping.")
        continue
        
    if is_pytorch:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = MLPClassifierNet(input_dim=X_test.shape[1], num_classes=4).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        evals = evaluate_model(name, model, X_test, y_test, is_pytorch=True, device=device)
        trained_models_eval[name] = (model, evals)
    else:
        model = joblib.load(model_path)
        # Fix LightGBM thread issues
        if name == "LightGBM":
            model.n_jobs = 1
        evals = evaluate_model(name, model, X_test, y_test, is_pytorch=False)
        trained_models_eval[name] = (model, evals)
        
    results_metrics.append({
        "Model": name,
        **{k: v for k, v in evals.items() if k not in ["probs", "preds"]}
    })

# 6. Save Cross-Dataset Comparison CSV
df_cross = pd.DataFrame(results_metrics)
df_cross.to_csv(os.path.join(BASE_DIR, "tables", "cross_dataset_results.csv"), index=False)
logger.info("Saved cross-dataset validation results table.")

# 7. Classification Reports compile
reports = []
for name, (_, evals) in trained_models_eval.items():
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
pd.DataFrame(reports).to_csv(os.path.join(BASE_DIR, "tables", "cross_dataset_classification_reports.csv"), index=False)
logger.info("Saved cross-dataset classification reports table.")

# 8. Visualizations
logger.info("Generating cross-dataset ROC and PR curves...")

# A. Multiclass ROC Curves
plt.figure(figsize=(8, 6))
for name, (_, evals) in trained_models_eval.items():
    probs = evals["probs"]
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    for i in range(len(CLASSES)):
        y_true_bin = (y_test == i).astype(int)
        y_prob_bin = probs[:, i]
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
plt.title('Macro-Average ROC Curves (Cross-Dataset: Test CICIDS2017)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "figures", "cross_dataset", "roc_curve.png"), dpi=300)
plt.close()

# B. Multiclass PR Curves
plt.figure(figsize=(8, 6))
for name, (_, evals) in trained_models_eval.items():
    probs = evals["probs"]
    mean_recall = np.linspace(0, 1, 100)
    precisions = []
    for i in range(len(CLASSES)):
        y_true_bin = (y_test == i).astype(int)
        y_prob_bin = probs[:, i]
        p_vals, r_vals, _ = precision_recall_curve(y_true_bin, y_prob_bin)
        precisions.append(np.interp(mean_recall, r_vals[::-1], p_vals[::-1]))
    mean_precision = np.mean(precisions, axis=0)
    macro_pr_auc = auc(mean_recall, mean_precision)
    plt.plot(mean_recall, mean_precision, label=f"{name} (PR AUC = {macro_pr_auc:.3f})", lw=2)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Macro-Average PR Curves (Cross-Dataset: Test CICIDS2017)')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "figures", "cross_dataset", "pr_curve.png"), dpi=300)
plt.close()

logger.info("Cross-Dataset Validation complete! All figures and reports saved successfully.")
