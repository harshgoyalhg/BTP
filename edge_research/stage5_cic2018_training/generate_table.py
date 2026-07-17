import pandas as pd

# Files
mc_val = pd.read_csv('tables/model_comparison.csv')
bin_val = pd.read_csv('tables/binary_results/binary_within_dataset_2018.csv')
mc_c17 = pd.read_csv('tables/cross_dataset_results.csv')
bin_c17 = pd.read_csv('tables/binary_results/binary_cross_dataset_2017.csv')
mc_ly = pd.read_csv('tables/lycos_results.csv')
bin_ly = pd.read_csv('tables/binary_results/binary_lycos_ext_results.csv')

models = ["Logistic Regression", "Random Forest", "Extra Trees", "XGBoost", "LightGBM", "MLP (PyTorch)", "SVM (Linear)", "SVM (RBF Kernel)"]

print("| Algorithm | Evaluation Scope | Test Data | Setup | Accuracy | F1 Score |")
print("|---|---|---|---|---|---|")

for m in models:
    # Within Multiclass
    row = mc_val[mc_val['Model'] == m]
    if len(row): print(f"| {m} | Within-Dataset | CICIDS-2018 | Multiclass (4-class) | {row['Accuracy'].values[0]:.4f} | {row['Weighted F1'].values[0]:.4f} |")
    
    # Within Binary
    row = bin_val[bin_val['Model'] == m]
    if len(row): print(f"| {m} | Within-Dataset | CICIDS-2018 | Binary (Benign vs Attack) | {row['Accuracy'].values[0]:.4f} | {row['Weighted F1'].values[0]:.4f} |")
    
    # Cross CIC17 Multiclass
    row = mc_c17[mc_c17['Model'] == m]
    if len(row): print(f"| {m} | Cross-Dataset | CICIDS-2017 | Multiclass (4-class) | {row['Accuracy'].values[0]:.4f} | {row['Weighted F1'].values[0]:.4f} |")
    
    # Cross CIC17 Binary
    row = bin_c17[bin_c17['Model'] == m]
    if len(row): print(f"| {m} | Cross-Dataset | CICIDS-2017 | Binary (Benign vs Attack) | {row['Accuracy'].values[0]:.4f} | {row['Weighted F1'].values[0]:.4f} |")
    
    # Cross Lycos Multiclass
    row = mc_ly[mc_ly['Model'] == m]
    if len(row): print(f"| {m} | Cross-Dataset | LycoS-2018 | Multiclass (4-class)* | {row['Accuracy'].values[0]:.4f} | {row['Weighted F1'].values[0]:.4f} |")
    
    # Cross Lycos Binary
    row = bin_ly[bin_ly['Model'] == m]
    if len(row): print(f"| {m} | Cross-Dataset | LycoS-2018 | Binary (Benign vs Attack) | {row['Accuracy'].values[0]:.4f} | {row['Weighted F1'].values[0]:.4f} |")

