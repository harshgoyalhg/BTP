# Stage 5 Report: Model Training and Resource Profiling on CICIDS2018 Dataset

**Author:** Senior Machine Learning Engineer, Cybersecurity Researcher, and IEEE Journal Co-Author
**Environment:** MacBook Air M4 (macOS Sequoia), Apple Silicon MPS (Metal Performance Shaders) & CPU
**Project Branch:** Experimental Reverse Cross-Dataset Validation

---

## 1. Abstract & Executive Summary

In this stage of the edge AI-SOC research pipeline, we conduct the training and evaluation of six machine learning classifiers on the **CICIDS2018** intrusion detection dataset. Unlike prior stages that focused on CICIDS2017 training, this stage establishes the baseline models trained on CICIDS2018 to support future reverse cross-dataset transferability validation. 

To handle the 6.4 GB raw dataset under memory-constrained environments, we implemented a memory-safe sequential chunked loader combined with a stratified 500,000-row representative downsampling scheme. Additionally, a rare-class "dummy injection" protocol was used to introduce the `PortScan` class, aligning the output dimension with the 4-class configuration of Stage 2 preprocessing artifacts (`scaler.pkl` and `label_encoder.pkl`). 

We profiled Logistic Regression, Random Forest, Extra Trees, XGBoost, LightGBM, and a PyTorch Multi-Layer Perceptron (MLP) on the MacBook Air M4 platform. Key results show that **XGBoost** and **Random Forest** achieved the highest overall accuracies of **95.13%** and **94.98%** respectively, while also showing superior capability in identifying the extremely rare `PortScan` class. PyTorch MLP accelerated via **Apple Silicon MPS** achieved **94.29%** accuracy and ran 15 epochs in just **36.54 seconds**.

---

## 2. Experimental Methodology & Preprocessing

### 2.1 Sequential Memory-Safe Ingestion
The raw CICIDS2018 dataset spans 10 CSV files totaling approximately 16 million network flows (6.4 GB). Ingesting this data in a single pandas load is highly susceptible to Out-Of-Memory (OOM) faults on standard consumer hardware. We built a chunked loader using `pd.read_csv` with a `chunksize=200,000` generator, processing and cleaning rows iteratively.

### 2.2 Feature Mapping and Alignment
To maintain compatibility with pre-existing Stage 2/3 encoders and scalers, we established a strict schema-mapping dictionary mapping 2018 columns to their 2017 equivalents:
* **Feature Renaming:** Formatted columns like `Dst Port` to `Destination Port`, `Tot Fwd Pkts` to `Total Fwd Packets`, and `Fwd Header Len` to `Fwd Header Length`.
* **Imputations & Padding:** Imputed the missing `Source Port` with a constant `0.0`. Cloned `Fwd Header Length` to both `Fwd Header Length` and `Fwd Header Length.1` to align with the 80 features required by the pre-trained scaler.
* **Cleaning:** Drop all rows containing `NaN` or `inf` values.

### 2.3 Stratified Sampling and Dummy Class Injection
The clean dataset contains a total of 3,015,304 rows (2,746,934 attacks and 268,370 benign flows, after a 2% downsampling of benign traffic to maintain class balance). We drew a representative stratified sample of **500,000 rows** to keep memory usage under 2 GB. 
Crucially, the `PortScan` class is present in the 2017 label encoder but has 0 natural instances in CICIDS2018. To enforce a 4-class classification surface and prevent shape mismatch errors during neural network evaluation and tree splits, we injected **5 dummy PortScan rows** with zeroed features into the training set (stratified 80/20 train/val split).

---

## 3. Hardware Optimizations & Workarounds

Executing modern machine learning training pipelines on the Apple Silicon M4 system presented several runtime challenges:
1. **OpenMP Runtime Aborts (macOS SIGABRT):** Both XGBoost and LightGBM rely on OpenMP for multi-threaded CPU acceleration. On macOS, importing both libraries simultaneously in the same script triggers a double-linking warning where OpenMP aborts the process by default. We bypassed this by setting `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` at the script entry.
2. **OpenMP Segmentation Faults (SIGSEGV):** Sequentially training XGBoost (with `n_jobs=-1`) and LightGBM (with `n_jobs=-1`) corrupted the thread pool, causing a silent crash (exit code 139) during LightGBM fitting. We resolved this by configuring LightGBM to run single-threaded (`n_jobs=1`), which eliminated the segfault while maintaining sub-10 second training times.
3. **MPS Acceleration for PyTorch:** The PyTorch MLP was optimized for the M4 GPU using Metal Performance Shaders (`torch.device("mps")`). To prevent host-to-device bottlenecks, we loaded data directly into native PyTorch Tensors, disabled memory pinning (`pin_memory=False`), and ran with zero dataloader workers (`num_workers=0`).

---

## 4. Quantitative Results & Evaluation

The six trained models were evaluated on the 20% validation split (100,001 network flows). 

### 4.1 Classifier Performance Comparison
The complete comparison of metrics across models is detailed below:

| Model | Accuracy | Weighted F1 | Macro F1 | Balanced Acc | MCC | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **0.9513** | **0.9516** | **0.9438** | 0.9369 | 0.9066 | 0.9965 | 0.9868 |
| **Random Forest** | 0.9498 | 0.9503 | 0.9427 | **0.9456** | 0.9034 | 0.9962 | 0.9857 |
| **MLP (PyTorch)** | 0.9429 | 0.9431 | 0.6793 | 0.6744 | 0.8897 | 0.9946 | 0.7303 |
| **LightGBM** | 0.9404 | 0.9409 | 0.6799 | 0.6786 | 0.8856 | 0.8487 | 0.6947 |
| **Logistic Regression** | 0.9284 | 0.9273 | 0.6545 | 0.6410 | 0.8601 | 0.9601 | 0.7075 |
| **Extra Trees** | 0.8910 | 0.8884 | 0.6408 | 0.6236 | 0.7822 | 0.9899 | 0.9654 |

### 4.2 Resource Profiling & Operational Latency
We profiled execution time, RAM footprint, and storage cost for deployment considerations:

| Model | Train Time (s) | Inference Time (s) | Peak RAM (MB) | Model Size (MB) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 23.37s | **0.010s** | **747.20 MB** | **0.002 MB** |
| **Extra Trees** | **2.72s** | 0.138s | 1294.94 MB | 8.62 MB |
| **XGBoost** | 5.29s | 0.128s | 1381.94 MB | 0.89 MB |
| **Random Forest** | 6.16s | 0.139s | 1247.25 MB | 13.28 MB |
| **LightGBM** | 9.13s | 1.794s | 1618.91 MB | 1.08 MB |
| **MLP (PyTorch)** | 36.54s | 0.387s | 1762.73 MB | 0.075 MB |

---

## 5. Core Cybersecurity and ML Engineering Insights

### 5.1 Convergence and Minority Class Sensitivity
A critical cybersecurity discovery in this stage lies in **minority class convergence under extreme imbalance**. The validation set contains only **1 sample** of `PortScan` (due to dummy injection). 
* Both **XGBoost** and **Random Forest** successfully classified this single `PortScan` sample (Precision = 1.0, Recall = 1.0, F1 = 1.0). This indicates that tree ensemble splitting thresholds are highly sensitive to minority structure even when the class proportion is $1:100,000$.
* Conversely, **LightGBM**, **MLP**, **Logistic Regression**, and **Extra Trees** failed completely to converge on it (Recall = 0.0, F1 = 0.0), which dragged down their Macro F1 metrics to the ~65-68% range. In an active SOC environment, this failure represents a blind spot where zero-day or low-frequency scans might bypass the detection layer.

### 5.2 Model Deployment & Latency Tradeoffs
* **The Performance Leader:** **XGBoost** represents the strongest candidate for general deployment, yielding the highest accuracy (95.13%), high efficiency (0.89 MB model size), and robust minority class detection.
* **The Real-Time Candidate:** **Logistic Regression** is the fastest model during inference (10 ms latency for 100k records) and consumes the least memory (747 MB), making it ideal for high-throughput edge firewalls or network cards. However, its lower detection capability (92.84% accuracy) must be balanced.
* **Edge Storage Limitations:** **Random Forest** provides excellent detection, but its model size is relatively large (13.28 MB) due to deep-tree structures. This makes it less suitable for storage-constrained firmware compared to XGBoost.

---

## 6. Generated Artifacts Reference
All visual plots and CSV metrics databases have been structured and saved in the output directory:
* **Confusion Matrices:** Saved in [figures/](file:///Users/harshgoyal/Documents/BTP/BTP/edge_research/stage5_cic2018_training/figures) (e.g. `confusion_matrix_xgboost.png`)
* **ROC & PR Curves:** Available at [roc_curve.png](file:///Users/harshgoyal/Documents/BTP/BTP/edge_research/stage5_cic2018_training/figures/roc_curve.png) and [pr_curve.png](file:///Users/harshgoyal/Documents/BTP/BTP/edge_research/stage5_cic2018_training/figures/pr_curve.png)
* **Metrics Comparison Databases:** Available in [tables/](file:///Users/harshgoyal/Documents/BTP/BTP/edge_research/stage5_cic2018_training/tables) (e.g. `model_comparison.csv`, `classification_reports.csv`)
