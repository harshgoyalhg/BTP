"""
config.py — Stage 6: CICIDS-2017 Training
==========================================
Single source of truth for paths, feature lists, hyperparameters,
label mappings, and class definitions.

All other scripts in stage6 import from this module.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = "/Users/harshgoyal/Documents/BTP/BTP/edge_research/stage6_cic2017_training"
DATASET_2017   = "/Users/harshgoyal/Documents/BTP/BTP/dataset/cicids-2017/TrafficLabelling "
STAGE5_DIR     = "/Users/harshgoyal/Documents/BTP/BTP/edge_research/stage5_cic2018_training"

ARTIFACT_DIR   = os.path.join(BASE_DIR, "artifacts")
MODEL_DIR      = os.path.join(BASE_DIR, "models")
TABLE_DIR      = os.path.join(BASE_DIR, "tables")
LOG_DIR        = os.path.join(BASE_DIR, "logs")
REPORT_DIR     = os.path.join(BASE_DIR, "reports")
FIG_DIR        = os.path.join(BASE_DIR, "figures")

# Stage 5 artifacts (used only for inverse-transform when doing cross-dataset)
STAGE5_ARTIFACT_DIR = os.path.join(STAGE5_DIR, "artifacts")

# ─── Classes ─────────────────────────────────────────────────────────────────
CLASSES     = ['BENIGN', 'Bot', 'DDoS', 'PortScan']
NUM_CLASSES = 4

# ─── Feature List (80 features — matches Stage 5 exactly) ────────────────────
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

INPUT_DIM_BASE = len(SCALER_FEATURES)          # 80
INPUT_DIM_ENG  = INPUT_DIM_BASE + 2            # 82 (after feature engineering)

# ─── CICIDS-2017 Label Mapping → 4 classes ───────────────────────────────────
# The 2017 dataset has richer label diversity than 2018 but collapses to the
# same 4-class schema for a fair, publication-ready comparison.
LABEL_MAPPING_2017 = {
    'BENIGN':                         'BENIGN',
    # PortScan — natively present in 2017 (no injection needed)
    'PortScan':                       'PortScan',
    # DDoS attacks
    'DDoS':                           'DDoS',
    'DoS Hulk':                       'DDoS',
    'DoS GoldenEye':                  'DDoS',
    'DoS slowloris':                  'DDoS',
    'DoS Slowhttptest':               'DDoS',
    'Heartbleed':                     'DDoS',    # protocol-level DoS exploit
    # Bot / Brute-force / Infiltration
    'Bot':                            'Bot',
    'FTP-Patator':                    'Bot',
    'SSH-Patator':                    'Bot',
    'Web Attack \u2013 Brute Force': 'Bot',
    'Web Attack \u2013 XSS':         'Bot',
    'Web Attack \u2013 Sql Injection':'Bot',
    'Infiltration':                   'Bot',
    # Handle any slight casing / encoding variants
    'Web Attack- Brute Force':        'Bot',
    'Web Attack- XSS':                'Bot',
    'Web Attack- Sql Injection':      'Bot',
}

# ─── Columns to drop from raw 2017 CSVs (non-feature metadata columns) ───────
COLS_TO_DROP = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']

# ─── Model Hyperparameters (IDENTICAL to Stage 5 — do not change) ─────────────
HP = {
    'logistic_regression': dict(C=1.0, max_iter=1000, random_state=42, class_weight=None),
    'random_forest':       dict(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42, class_weight=None),
    'extra_trees':         dict(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42, class_weight=None),
    'xgboost':             dict(n_estimators=100, max_depth=6, learning_rate=0.1,
                                tree_method='hist', device='cpu', n_jobs=-1, random_state=42),
    'lightgbm':            dict(n_estimators=100, max_depth=6, learning_rate=0.1,
                                n_jobs=1, random_state=42, verbose=-1, class_weight=None),
    'mlp': dict(
        hidden_1=128, hidden_2=64, dropout=0.2,
        lr=0.002, epochs=30, batch_size=512,
        scheduler='CosineAnnealingLR', T_max=30,
        loss='CrossEntropyLoss',
    ),
    'svm_linear': dict(C=0.1, dual=False, random_state=42, class_weight=None),
    'svm_rbf':    dict(C=1.0, kernel='rbf', probability=True, random_state=42, class_weight=None),
    # Downsampling sizes for SVMs (same as Stage 5)
    'svm_linear_n': 100_000,
    'svm_rbf_n':     20_000,
}

# ─── Data Sampling (identical to Stage 5) ────────────────────────────────────
STRATIFIED_SAMPLE_SIZE = 500_000
BENIGN_DOWNSAMPLE_FRAC = 0.02     # keep 2% of benign per chunk
CHUNK_SIZE = 200_000
TRAIN_TEST_SPLIT_RATIO = 0.2      # 80/20 split
RANDOM_STATE = 42

# ─── Engineering feature indices (after SCALER_FEATURES ordering) ─────────────
IDX_FWD_PKTS = SCALER_FEATURES.index('Total Fwd Packets')
IDX_BWD_PKTS = SCALER_FEATURES.index('Total Backward Packets')
IDX_FWD_LEN  = SCALER_FEATURES.index('Total Length of Fwd Packets')
IDX_BWD_LEN  = SCALER_FEATURES.index('Total Length of Bwd Packets')
