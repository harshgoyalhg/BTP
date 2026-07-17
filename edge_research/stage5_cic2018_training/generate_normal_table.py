import pandas as pd
import os

base = 'results_before_reoptimization/tables/'
mc_val = pd.read_csv(os.path.join(base, 'model_comparison.csv'))
mc_c17 = pd.read_csv(os.path.join(base, 'cross_dataset_results.csv'))
mc_ly = pd.read_csv(os.path.join(base, 'lycos_results.csv'))

models = ["Logistic Regression", "Random Forest", "Extra Trees", "XGBoost", "LightGBM", "MLP (PyTorch)"]

print("| Algorithm | Evaluation Scope | Test Data | Setup | Accuracy | F1 Score |")
print("|---|---|---|---|---|---|")

for m in models:
    # Within Multiclass
    row = mc_val[mc_val['Model'] == m]
    if len(row): print(f"| {m} | Within-Dataset | CICIDS-2018 | Multiclass (4-class) | {row['Accuracy'].values[0]:.4f} | {row['Weighted F1'].values[0]:.4f} |")
    
    # Cross CIC17 Multiclass
    row = mc_c17[mc_c17['Model'] == m]
    if len(row): print(f"| {m} | Cross-Dataset | CICIDS-2017 | Multiclass (4-class) | {row['Accuracy'].values[0]:.4f} | {row['Weighted F1'].values[0]:.4f} |")
    
    # Cross Lycos Multiclass
    row = mc_ly[mc_ly['Model'] == m]
    if len(row): print(f"| {m} | Cross-Dataset | LycoS-2018 | Multiclass (4-class)* | {row['Accuracy'].values[0]:.4f} | {row['Weighted F1'].values[0]:.4f} |")

