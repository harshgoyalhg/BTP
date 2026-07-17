import pandas as pd

df = pd.read_csv('tables/variable_features/variable_features_results.csv')

# Filter for Top 20 features (which is the peak performance mentioned)
df_20 = df[df['Num_Features'] == 20]

print("| Algorithm | Features | Evaluation Scope | Test Data | Setup | Accuracy | F1 Score |")
print("|---|---|---|---|---|---|---|")

def map_dataset(ds):
    if "CICIDS-2018" in ds: return "Within-Dataset", "CICIDS-2018"
    if "CICIDS-2017" in ds: return "Cross-Dataset", "CICIDS-2017"
    if "LycoS" in ds: return "Cross-Dataset", "LycoS-2018"
    return "Unknown", ds

def map_eval(ev, ds):
    if ev == "Multiclass" and "LycoS" in ds:
        return "Multiclass (4-class)*"
    if ev == "Multiclass":
        return "Multiclass (4-class)"
    return "Binary (Benign vs Attack)"

for _, row in df_20.iterrows():
    scope, td = map_dataset(row['Dataset'])
    setup = map_eval(row['Evaluation_Type'], row['Dataset'])
    print(f"| {row['Model']} | Top 20 | {scope} | {td} | {setup} | {row['Accuracy']:.4f} | {row['F1_Score']:.4f} |")

