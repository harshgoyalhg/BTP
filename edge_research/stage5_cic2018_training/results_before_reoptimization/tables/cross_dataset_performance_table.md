# Performance Metrics (Within-Dataset & Cross-Dataset Configurations)

| Train set | Test set | Classifier | MCC | F1 (Macro) | F1 (Weighted) | AUROC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **CIC18** | **CIC18 (Val)** | LR | 83.96% | 89.67% | 91.40% | 98.70% |
| | | RF | 88.06% | 91.92% | 93.60% | 99.51% |
| | | ET | 84.76% | 90.51% | 91.75% | 98.84% |
| | | XGB | 89.02% | 92.77% | 94.12% | 99.56% |
| | | LGBM | 89.35% | 93.04% | 94.30% | 99.59% |
| | | MLP | 87.96% | 91.98% | 93.55% | 99.46% |
| **CIC18** | **CIC17 (Test)** | LR | 49.56% | 48.65% | 66.47% | 82.55% |
| | | RF | 61.17% | 44.12% | 70.81% | 86.90% |
| | | ET | 63.58% | 45.07% | 72.34% | 87.88% |
| | | XGB | 50.13% | 44.60% | 67.57% | 78.43% |
| | | LGBM | 11.47% | 23.04% | 47.03% | 66.86% |
| | | MLP | 56.33% | 45.63% | 70.39% | 81.12% |
| **CIC18** | **LycoS18** | LR | -4.77% | 17.99% | 36.75% | N/A* |
| | | RF | 95.86% | 92.82% | 97.87% | N/A* |
| | | ET | 75.46% | 79.69% | 85.87% | N/A* |
| | | XGB | 73.42% | 70.25% | 85.98% | N/A* |
| | | LGBM | 87.98% | 83.16% | 94.00% | N/A* |
| | | MLP | 12.48% | 40.27% | 38.54% | N/A* |

*\*Note on LycoS18 AUROC: The LycoS-Unicas-IDS2018 dataset naturally contains only 3 active classes (it completely lacks the PortScan class). Therefore, standard multiclass One-vs-Rest (OvR) AUROC calculations are mathematically undefined (N/A) because the true label vector contains 0 occurrences of the fourth target class, throwing a validation warning.*
