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
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    balanced_accuracy_score, matthews_corrcoef
)
from sklearn.model_selection import train_test_split

# 1. Setup Directories & Logging
BASE_DIR = "/Users/harshgoyal/Documents/BTP/BTP/edge_research/stage5_cic2018_training"
TABLE_DIR = os.path.join(BASE_DIR, "tables", "variable_features")
FIG_DIR = os.path.join(BASE_DIR, "figures", "variable_features")
LOG_DIR = os.path.join(BASE_DIR, "logs")

for d in [TABLE_DIR, FIG_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "variable_features.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("variable_features")
logger.info("=" * 70)
logger.info("Stage 5 — Variable Features Evaluation")
logger.info("=" * 70)

CLASSES = ['BENIGN', 'Bot', 'DDoS', 'PortScan']
NUM_CLASSES = 4

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
FEATURE_NAMES = SCALER_FEATURES + ['Packet Ratio', 'Length Ratio']

class MLPClassifierNet(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# 2. Data Loading & Feature Engineering
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

logger.info("Loading CICIDS-2018 Train & Validation Sets...")
data_18 = np.load(os.path.join(BASE_DIR, "artifacts", "cicids2018_processed.npz"), allow_pickle=True)
X_train_80 = data_18['X_train'].astype(np.float32)
y_train = data_18['y_train'].astype(np.int64)
X_val_80 = data_18['X_val'].astype(np.float32)
y_val = data_18['y_val'].astype(np.int64)

# Remove dummy PortScan (label 3)
t_mask = (y_train != 3); X_train_80 = X_train_80[t_mask]; y_train = y_train[t_mask]
v_mask = (y_val != 3); X_val_80 = X_val_80[v_mask]; y_val = y_val[v_mask]

# Inject real PortScan from 2017
logger.info("Injecting PortScan from CICIDS-2017...")
data_17 = np.load(os.path.join(BASE_DIR, "artifacts", "cicids2017_processed.npz"))
ps_idx = np.where(data_17['y_train'] == 3)[0]
np.random.seed(42)
sel = np.random.choice(ps_idx, size=10000, replace=False)
X_ps = data_17['X_train'][sel]
y_ps = data_17['y_train'][sel]
X_ps_tr, X_ps_v, y_ps_tr, y_ps_v = train_test_split(X_ps, y_ps, test_size=0.2, random_state=42, stratify=y_ps)

X_train_80 = np.vstack([X_train_80, X_ps_tr]); y_train = np.concatenate([y_train, y_ps_tr])
X_val_80 = np.vstack([X_val_80, X_ps_v]); y_val = np.concatenate([y_val, y_ps_v])

X_train_82 = apply_feature_engineering(X_train_80)
X_val_82 = apply_feature_engineering(X_val_80)

logger.info("Loading CICIDS-2017 Test Set (Cross-Dataset)...")
X_cic17_80 = data_17['X_test'].astype(np.float32)
y_cic17 = data_17['y_test'].astype(np.int64)
X_cic17_82 = apply_feature_engineering(X_cic17_80)

logger.info("Loading LycoS Test Set (Cross-Dataset)...")
data_lycos = np.load(os.path.join(BASE_DIR, "artifacts", "lycos_processed.npz"))
X_lycos_80 = data_lycos['X_test'].astype(np.float32)
y_lycos_mc = data_lycos['y_test'].astype(np.int64)
X_lycos_82 = apply_feature_engineering(X_lycos_80)

logger.info(f"Train Shape: {X_train_82.shape}")

# 3. Determine Feature Importance
rf_model_path = os.path.join(BASE_DIR, "models", "random_forest.pkl")
if os.path.exists(rf_model_path):
    logger.info("Loading pre-trained Random Forest to get feature importances...")
    base_rf = joblib.load(rf_model_path)
    importances = base_rf.feature_importances_
else:
    logger.info("Pre-trained RF not found. Training a quick RF to get importances...")
    base_rf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    base_rf.fit(X_train_82, y_train)
    importances = base_rf.feature_importances_

sorted_indices = np.argsort(importances)[::-1]
logger.info(f"Top 10 features: {[FEATURE_NAMES[i] for i in sorted_indices[:10]]}")

# 4. Evaluation Loop
def evaluate_metrics(model, X, y_true, is_pytorch=False, device='cpu'):
    if is_pytorch:
        model.eval()
        with torch.no_grad():
            outputs = model(torch.tensor(X, dtype=torch.float32).to(device))
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            y_pred = np.argmax(probs, axis=1)
    else:
        y_pred = model.predict(X)

    # Multiclass Metrics
    mc_acc = accuracy_score(y_true, y_pred)
    mc_f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Binary Metrics (0: Benign, 1: Attack)
    y_true_bin = (y_true != 0).astype(int)
    y_pred_bin = (y_pred != 0).astype(int)
    bin_acc = accuracy_score(y_true_bin, y_pred_bin)
    bin_f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    
    return mc_acc, mc_f1, bin_acc, bin_f1

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
feature_counts = [10, 20, 30, 40, 50, 60, 70, 82]

results = []

for N in feature_counts:
    logger.info(f"\n{'='*40}\nRunning experiments with Top {N} features\n{'='*40}")
    selected_features = sorted_indices[:N]
    
    X_train_subset = X_train_82[:, selected_features]
    X_val_subset = X_val_82[:, selected_features]
    X_cross_subset = X_cic17_82[:, selected_features]
    X_lycos_subset = X_lycos_82[:, selected_features]
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, tree_method="hist", device="cpu", n_jobs=-1, random_state=42),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=1, random_state=42, verbose=-1),
        "MLP (PyTorch)": MLPClassifierNet(input_dim=N, num_classes=4).to(device)
    }
    
    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        t0 = time.time()
        is_pt = (model_name == "MLP (PyTorch)")
        
        if is_pt:
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.002)
            train_dataset = TensorDataset(torch.tensor(X_train_subset, dtype=torch.float32), torch.tensor(y_train, dtype=torch.int64))
            train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
            model.train()
            for epoch in range(15): # Reduced to 15 epochs for speed during this feature sweep
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
        else:
            model.fit(X_train_subset, y_train)
            
        train_time = time.time() - t0
        
        # Evaluate
        mc_acc_18, mc_f1_18, bin_acc_18, bin_f1_18 = evaluate_metrics(model, X_val_subset, y_val, is_pytorch=is_pt, device=device)
        mc_acc_17, mc_f1_17, bin_acc_17, bin_f1_17 = evaluate_metrics(model, X_cross_subset, y_cic17, is_pytorch=is_pt, device=device)
        mc_acc_lycos, mc_f1_lycos, bin_acc_lycos, bin_f1_lycos = evaluate_metrics(model, X_lycos_subset, y_lycos_mc, is_pytorch=is_pt, device=device)
        
        results.append({
            "Num_Features": N,
            "Model": model_name,
            "Train_Time_s": round(train_time, 2),
            "Dataset": "CICIDS-2018 (Within)",
            "Evaluation_Type": "Multiclass",
            "Accuracy": round(mc_acc_18, 4),
            "F1_Score": round(mc_f1_18, 4)
        })
        results.append({
            "Num_Features": N,
            "Model": model_name,
            "Train_Time_s": round(train_time, 2),
            "Dataset": "CICIDS-2018 (Within)",
            "Evaluation_Type": "Binary",
            "Accuracy": round(bin_acc_18, 4),
            "F1_Score": round(bin_f1_18, 4)
        })
        results.append({
            "Num_Features": N,
            "Model": model_name,
            "Train_Time_s": round(train_time, 2),
            "Dataset": "CICIDS-2017 (Cross)",
            "Evaluation_Type": "Multiclass",
            "Accuracy": round(mc_acc_17, 4),
            "F1_Score": round(mc_f1_17, 4)
        })
        results.append({
            "Num_Features": N,
            "Model": model_name,
            "Train_Time_s": round(train_time, 2),
            "Dataset": "CICIDS-2017 (Cross)",
            "Evaluation_Type": "Binary",
            "Accuracy": round(bin_acc_17, 4),
            "F1_Score": round(bin_f1_17, 4)
        })
        results.append({
            "Num_Features": N,
            "Model": model_name,
            "Train_Time_s": round(train_time, 2),
            "Dataset": "LycoS-2018 (Cross)",
            "Evaluation_Type": "Multiclass",
            "Accuracy": round(mc_acc_lycos, 4),
            "F1_Score": round(mc_f1_lycos, 4)
        })
        results.append({
            "Num_Features": N,
            "Model": model_name,
            "Train_Time_s": round(train_time, 2),
            "Dataset": "LycoS-2018 (Cross)",
            "Evaluation_Type": "Binary",
            "Accuracy": round(bin_acc_lycos, 4),
            "F1_Score": round(bin_f1_lycos, 4)
        })

df = pd.DataFrame(results)
df.to_csv(os.path.join(TABLE_DIR, "variable_features_results.csv"), index=False)
logger.info("Results saved to CSV.")

# 5. Plotting
logger.info("Generating plots...")

def plot_metric(df_plot, title, filename):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_plot, x="Num_Features", y="Accuracy", hue="Model", marker="o", lw=2)
    plt.title(title)
    plt.xlabel("Number of Features Used")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300)
    plt.close()

plot_metric(df[(df["Dataset"] == "CICIDS-2018 (Within)") & (df["Evaluation_Type"] == "Multiclass")], 
            "Multiclass Accuracy vs Num Features (Within CICIDS-2018)", "multiclass_within_2018.png")
plot_metric(df[(df["Dataset"] == "CICIDS-2018 (Within)") & (df["Evaluation_Type"] == "Binary")], 
            "Binary Accuracy vs Num Features (Within CICIDS-2018)", "binary_within_2018.png")

plot_metric(df[(df["Dataset"] == "CICIDS-2017 (Cross)") & (df["Evaluation_Type"] == "Multiclass")], 
            "Multiclass Accuracy vs Num Features (Cross CICIDS-2017)", "multiclass_cross_2017.png")
plot_metric(df[(df["Dataset"] == "CICIDS-2017 (Cross)") & (df["Evaluation_Type"] == "Binary")], 
            "Binary Accuracy vs Num Features (Cross CICIDS-2017)", "binary_cross_2017.png")

logger.info("Variable Features Evaluation COMPLETE.")
