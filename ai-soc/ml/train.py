import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import os
import time

def train_model():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")

    print("Loading processed dataset...")
    data = np.load(os.path.join(data_dir, "cicids2017_processed.npz"))
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")

    # Load label encoder to get class names
    le = joblib.load(os.path.join(data_dir, "label_encoder.pkl"))
    class_names = le.classes_
    print(f"Classes to predict: {class_names}")

    print("\nInitializing XGBoost Classifier (with GPU support)...")
    # Using 'hist' with 'cuda' device is the modern way for GPU training in XGBoost >= 2.0
    # For older versions, tree_method='gpu_hist' would be used. We'll use the modern params.
    try:
        clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            tree_method='hist',
            device='cuda', # Force GPU training
            n_jobs=-1,
            random_state=42
        )
    except Exception as e:
        print(f"Warning: Could not initialize with modern GPU parameters. Error: {e}")
        clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            tree_method='gpu_hist',
            random_state=42
        )

    print("Training model (this might take a moment depending on your GPU)...")
    start_time = time.time()
    clf.fit(X_train, y_train)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    print("\nMaking predictions on test set...")
    y_pred = clf.predict(X_test)

    print("\n" + "="*50)
    print("MODEL EVALUATION METRICS")
    print("="*50)

    # Calculate metrics using 'weighted' average since it's a multi-class problem
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    
    # Print formatted confusion matrix
    print(f"{'':>12} " + " ".join([f"{c[:10]:>10}" for c in class_names]))
    for i, row in enumerate(cm):
        print(f"{class_names[i][:10]:>12} " + " ".join([f"{val:>10}" for val in row]))

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    print(f"\nSaving model to {model_path}...")
    joblib.dump(clf, model_path)
    print("Model saved successfully!")

if __name__ == "__main__":
    train_model()
