# Stage 6: Variable Features Evaluation (CICIDS-2017)

## Overview
To analyze how feature set size impacts model performance, we conducted an ablation study on the CICIDS-2017 dataset where models were trained and evaluated on subsets of the top $N$ most important features ($N \in \{10, 20, 30, 40, 50, 60, 70, 82\}$). Feature importance was derived using the pre-trained Random Forest baseline.

* The models were re-trained from scratch for each feature subset while keeping hyperparameters strictly constant.
* Both Multiclass (4-class) and Binary (Benign vs. Attack) metrics were evaluated for within-dataset (CICIDS-2017) and cross-dataset (CICIDS-2018) scenarios.
* The results are stored in isolated folders for easy identification.

## Observations
* **Robustness of Trees:** Tree-based models (XGBoost, Random Forest, LightGBM) maintain high performance even when restricted to the top 20-30 features, suggesting significant feature redundancy in the full 82-feature vector for the CICIDS-2017 data as well.
* **Top 10 Features:** The most important features dictating the tree splits were: `'Destination Port', 'Bwd Packet Length Min', 'Bwd Packets/s', 'Total Fwd Packets', 'Fwd Header Length.1', 'Subflow Fwd Packets', 'Fwd Header Length', 'Flow Bytes/s', 'Init_Win_bytes_forward', 'Subflow Fwd Bytes'`.

## Artifacts
* **Execution Script:** `scripts/evaluate_variable_features.py`
* **Performance Charts:** `figures/variable_features/`
* **Raw CSV Results:** `tables/variable_features/variable_features_results.csv`
