import os
import joblib
import numpy as np
import xgboost as xgb

# Define paths
ML_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(ML_DIR, "..", "data", "processed")
MODEL_PATH = os.path.join(ML_DIR, "model.pkl")
SCALER_PATH = os.path.join(PROCESSED_DATA_DIR, "scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(PROCESSED_DATA_DIR, "label_encoder.pkl")

# Global variables to hold our loaded models so they are only loaded once
_model = None
_scaler = None
_label_encoder = None

def _load_artifacts():
    """Lazily load the model, scaler, and label encoder."""
    global _model, _scaler, _label_encoder
    
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")
        _model = joblib.load(MODEL_PATH)
        
    if _scaler is None:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}.")
        _scaler = joblib.load(SCALER_PATH)
        
    if _label_encoder is None:
        if not os.path.exists(LABEL_ENCODER_PATH):
            raise FileNotFoundError(f"Label Encoder not found at {LABEL_ENCODER_PATH}.")
        _label_encoder = joblib.load(LABEL_ENCODER_PATH)

def predict_attack(features):
    """
    Predicts the attack class and confidence score for a given set of features.
    
    Args:
        features (list or numpy.ndarray): A single row of features (length 80) 
                                          or multiple rows for batch prediction.
                                          
    Returns:
        dict: A dictionary containing 'attack_name' and 'confidence_score' 
              (or a list of dicts if multiple rows are provided).
    """
    _load_artifacts()
    
    # Convert input to numpy array
    features_array = np.array(features)
    
    # Reshape if a single sample (1D array) is provided
    single_prediction = False
    if features_array.ndim == 1:
        features_array = features_array.reshape(1, -1)
        single_prediction = True
        
    # Standardize the features using the scaler fitted during preprocessing
    # IMPORTANT: The model was trained on scaled data!
    features_scaled = _scaler.transform(features_array)
    
    # Predict probabilities to calculate confidence score
    probabilities = _model.predict_proba(features_scaled)
    
    # Get the predicted class indices and their max probabilities
    predicted_indices = np.argmax(probabilities, axis=1)
    confidence_scores = np.max(probabilities, axis=1)
    
    # Decode the numerical indices back to attack names (e.g., 'BENIGN', 'DDoS')
    attack_names = _label_encoder.inverse_transform(predicted_indices)
    
    # Format the results
    results = []
    for name, conf in zip(attack_names, confidence_scores):
        results.append({
            "attack_name": str(name),
            "confidence_score": float(conf)
        })
        
    if single_prediction:
        return results[0]
    return results

if __name__ == "__main__":
    # Simple test to ensure the prediction engine works
    print("Testing Prediction Engine...")
    try:
        # Load the test set just to grab a sample for testing
        data = np.load(os.path.join(PROCESSED_DATA_DIR, "cicids2017_processed.npz"))
        X_test = data['X_test'] # Note: This is already scaled, but we'll use a dummy unscaled vector to test the pipeline logic
        y_test = data['y_test']
        
        # We'll create a dummy array of zeros matching the feature count (80 features)
        dummy_features = np.zeros(80)
        
        print("\nTesting single prediction...")
        result = predict_attack(dummy_features)
        print(f"Prediction result: {result}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
