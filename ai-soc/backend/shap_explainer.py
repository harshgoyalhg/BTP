import shap
import joblib
import numpy as np
import os

ML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml")
MODEL_PATH = os.path.join(ML_DIR, "model.pkl")

# Note: CICIDS2017 features after dropping 'Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Label'
FEATURE_NAMES = [
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
    'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]

_model = None
_explainer = None

def _load_explainer():
    global _model, _explainer
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _explainer is None:
        # TreeExplainer is super fast for XGBoost
        _explainer = shap.TreeExplainer(_model)

def get_shap_values(scaled_features_array):
    """
    Returns the top contributing features for the prediction.
    """
    _load_explainer()
    
    # SHAP values for the specific prediction
    shap_values = _explainer.shap_values(scaled_features_array)
    
    # shap_values might be a list (one array per class) for multiclass, 
    # or a 3D array depending on XGBoost and SHAP versions.
    # We will compute the absolute mean contribution across all classes to find the most important features.
    
    if isinstance(shap_values, list):
        # sum abs values across all classes to find overall impact
        impact = np.sum([np.abs(sv[0]) for sv in shap_values], axis=0)
    else:
        # If it's a 3D array (num_samples, num_features, num_classes)
        if len(shap_values.shape) == 3:
            impact = np.sum(np.abs(shap_values[0]), axis=1)
        else:
            impact = np.abs(shap_values[0])

    # Combine with feature names
    feature_impacts = {}
    for i, name in enumerate(FEATURE_NAMES):
        # Safely handle if we have fewer names than features
        if i < len(impact):
            feature_impacts[name] = float(impact[i])
            
    # Sort and return top 5
    sorted_impacts = dict(sorted(feature_impacts.items(), key=lambda item: item[1], reverse=True)[:5])
    return sorted_impacts
