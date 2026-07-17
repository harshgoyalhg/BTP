import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = "/Users/harshgoyal/Documents/BTP/BTP/edge_research/stage5_cic2018_training"

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleMLP, self).__init__()
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

# Load Transformers
scaler = joblib.load(os.path.join(BASE_DIR, "artifacts", "scaler.pkl"))
engineered_scaler = joblib.load(os.path.join(BASE_DIR, "artifacts", "engineered_scaler.pkl"))

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

def load_data():
    print("Loading datasets...")
    data_18 = np.load(os.path.join(BASE_DIR, "artifacts", "cicids2018_processed.npz"), allow_pickle=True)
    X_18 = data_18['X_val'].astype(np.float32)
    y_18 = data_18['y_val'].astype(np.int64)
    
    data_17 = np.load(os.path.join(BASE_DIR, "artifacts", "cicids2017_processed.npz"))
    X_17 = data_17['X_test'].astype(np.float32)
    y_17 = data_17['y_test'].astype(np.int64)
    
    # Load Lycos
    data_lycos = np.load(os.path.join(BASE_DIR, "artifacts", "lycos_processed.npz"))
    X_ly = data_lycos["X_test"].astype(np.float32)
    y_ly = data_lycos["y_test"].astype(np.int64)
    
    return (X_18, y_18), (X_17, y_17), (X_ly, y_ly)

def get_metrics(model_name, model, X, y, is_mlp=False):
    # Remove PortScan (Label 3) from true data
    mask = (y != 3)
    X_sub = X[mask]
    y_sub = y[mask]
    
    if is_mlp:
        model.eval()
        with torch.no_grad():
            tensor_X = torch.tensor(X_sub, dtype=torch.float32)
            out = model(tensor_X)
            preds = torch.argmax(out, dim=1).numpy()
    else:
        preds = model.predict(X_sub)
    
    # Multiclass 3-class F1 over labels [0,1,2] only
    mc_acc = accuracy_score(y_sub, preds)
    mc_f1 = f1_score(y_sub, preds, average='weighted', labels=[0,1,2])
    
    # Binary: Benign(0) vs Attack(!=0)
    y_sub_bin = (y_sub != 0).astype(int)
    preds_bin = (preds != 0).astype(int)
    bin_acc = accuracy_score(y_sub_bin, preds_bin)
    bin_f1 = f1_score(y_sub_bin, preds_bin, average='weighted')
    
    return mc_acc, mc_f1, bin_acc, bin_f1

(X_18, y_18), (X_17, y_17), (X_ly, y_ly) = load_data()

X_18_82 = apply_feature_engineering(X_18)
X_17_82 = apply_feature_engineering(X_17)
X_ly_82 = apply_feature_engineering(X_ly)

models = {
    'Logistic Regression': joblib.load('../models/logistic_regression.pkl'),
    'Random Forest': joblib.load('../models/random_forest.pkl'),
    'Extra Trees': joblib.load('../models/extra_trees.pkl'),
    'XGBoost': joblib.load('../models/xgboost.pkl'),
    'LightGBM': joblib.load('../models/lightgbm.pkl')
}
mlp = SimpleMLP(82, 4)
mlp.load_state_dict(torch.load('../models/mlp.pt', map_location='cpu'))
models['MLP (PyTorch)'] = mlp

results = []
for name, model in models.items():
    print(f"Evaluating {name}...")
    is_mlp = (name == 'MLP (PyTorch)')
    
    mc_acc18, mc_f118, bin_acc18, bin_f118 = get_metrics(name, model, X_18_82, y_18, is_mlp)
    mc_acc17, mc_f117, bin_acc17, bin_f117 = get_metrics(name, model, X_17_82, y_17, is_mlp)
    mc_accLy, mc_f1Ly, bin_accLy, bin_f1Ly = get_metrics(name, model, X_ly_82, y_ly, is_mlp)
    
    # Multiclass rows
    results.append(f"| {name} | Within-Dataset | CICIDS-2018 | 3-Class (No PortScan) | {mc_acc18:.4f} | {mc_f118:.4f} |")
    results.append(f"| {name} | Within-Dataset | CICIDS-2018 | Binary (No PortScan) | {bin_acc18:.4f} | {bin_f118:.4f} |")
    results.append(f"| {name} | Cross-Dataset | CICIDS-2017 | 3-Class (No PortScan) | {mc_acc17:.4f} | {mc_f117:.4f} |")
    results.append(f"| {name} | Cross-Dataset | CICIDS-2017 | Binary (No PortScan) | {bin_acc17:.4f} | {bin_f117:.4f} |")
    results.append(f"| {name} | Cross-Dataset | LycoS-2018 | 3-Class (No PortScan) | {mc_accLy:.4f} | {mc_f1Ly:.4f} |")
    results.append(f"| {name} | Cross-Dataset | LycoS-2018 | Binary (No PortScan) | {bin_accLy:.4f} | {bin_f1Ly:.4f} |")

print("\n| Algorithm | Evaluation Scope | Test Data | Setup | Accuracy | F1 Score |")
print("|---|---|---|---|---|---|")
for r in results:
    print(r)
