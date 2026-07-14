# Performance Metrics (Within-Dataset & Cross-Dataset Configurations)

| Train set | Test set | Classifier | MCC | F1 (Macro) | F1 (Weighted) | AUROC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **CIC18** | **CIC18 (Val)** | LR | 86.52% | 89.73% | 92.71% | 98.81% |
| | | RF | 90.90% | 94.29% | 95.14% | 99.63% |
| | | ET | 79.27% | 89.18% | 88.99% | 99.09% |
| | | XGB | 91.10% | 94.33% | 95.23% | 99.65% |
| | | LGBM | 91.40% | 94.64% | 95.39% | 99.67% |
| | | MLP | 89.96% | 93.22% | 94.63% | 99.56% |
| | | SVM (Linear)| 85.98% | 89.30% | 92.41% | 98.38% |
| | | SVM (RBF) | 62.70% | 56.29% | 79.87% | 96.39% |
| **CIC18** | **CIC17 (Test)**| LR | 52.51% | 48.47% | 57.28% | 82.49% |
| | | RF | 22.65% | 22.17% | 35.84% | 75.48% |
| | | ET | 53.32% | 46.22% | 68.96% | 85.40% |
| | | XGB | 18.06% | 16.99% | 25.34% | 88.20% |
| | | LGBM | 5.17% | 9.91% | 9.88% | 67.06% |
| | | MLP | 50.39% | 39.56% | 65.23% | 76.33% |
| | | SVM (Linear)| 51.77% | 49.50% | 57.98% | 84.42% |
| | | SVM (RBF) | 5.47% | 20.72% | 35.69% | 81.70% |
| **CIC18** | **LycoS18** | LR | -24.68% | 12.62% | 25.69% | N/A* |
| | | RF | 67.10% | 64.29% | 79.01% | N/A* |
| | | ET | 54.29% | 52.81% | 64.93% | N/A* |
| | | XGB | 28.84% | 42.39% | 51.96% | N/A* |
| | | LGBM | 62.06% | 59.10% | 72.73% | N/A* |
| | | MLP | 30.29% | 30.21% | 56.09% | N/A* |
| | | SVM (Linear)| 4.40% | 10.25% | 18.24% | N/A* |
| | | SVM (RBF) | 33.91% | 36.07% | 49.28% | N/A* |

*\*Note on LycoS18 AUROC: The LycoS-Unicas-IDS2018 dataset naturally contains only 3 active classes (it completely lacks the PortScan class). Therefore, standard multiclass One-vs-Rest (OvR) AUROC calculations are mathematically undefined (N/A) because the true label vector contains 0 occurrences of the fourth target class.*
