# Kaggle Writeup — Network Intrusion Detection System (UNSW-NB15)

## Overview

This project builds an ML-based network intrusion detection system using the UNSW-NB15 dataset. The goal is to solve two Kaggle tasks:

- **Task 1:** Binary classification of network flows as Normal or Attack
- **Task 2:** Multi-class classification of attack family labels

The `k_security_v2.ipynb` notebook contains the latest v2 pipeline, including hierarchical cascading, ensemble modelling, ADASYN oversampling, and per-class threshold tuning.

## Dataset

The dataset includes:

- Training set: 175,341 records
- Testing set: 82,332 records
- 34 original numeric/categorical features
- Targets:
  - `label` (binary): 0 = Normal, 1 = Attack
  - `attack_cat` (multi-class): Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms

## Problem Statement

Signature-based intrusion detection struggles with novel attacks and polymorphic traffic. The objective is to learn statistical patterns in flows so the model can generalise to unseen malicious traffic.

### Task 1 — Binary
Predict whether a flow is normal or malicious. The main evaluation metric is **F1 Score**.

### Task 2 — Multi-Class
Predict the attack family for malicious flows. The main evaluation metric is **Macro F1 Score**, which rewards balanced performance across all attack categories.

## v2 Methodology

### 1. Preprocessing
- Load and clean UNSW-NB15 training and testing data from CSV/parquet
- Drop unnamed index columns
- Encode categorical features: `proto`, `service`, `state`
- Standard-scale numeric features
- Generate engineered features such as ratio and log transforms of flow statistics

### 2. Imbalance handling
- Use **ADASYN** to create harder synthetic examples for minority classes
- Fall back to **SMOTE** when ADASYN is unstable for a given split
- Apply oversampling separately for binary and multi-class training

### 3. Modelling approaches

#### Approach A — Hierarchical cascade
- Stage 1: binary XGBoost gate predicts attack probability
- Stage 2: multi-class classifier runs only on flows predicted as attacks
- Cascade aims to reduce noise for multi-class prediction and focus on attack families

#### Approach B — Flat multi-class models
- Train models directly on all 10 classes
- Models include:
  - XGBoost multi-class
  - LightGBM multi-class
  - CatBoost multi-class
  - Stacking ensemble combining LGBM, XGB, and CatBoost with a Logistic Regression meta-learner

#### Threshold tuning
- Binary classifier threshold is optimised for F1 rather than using 0.5
- Multi-class models receive per-class probability threshold tuning to maximise Macro F1

## Evaluation

### Task 1
- Binary XGBoost is the baseline and submission model
- Metrics include F1, precision, recall, and ROC-AUC
- Threshold tuning improves decision boundaries for the attack class

### Task 2
- Compare flat models vs cascade strategy
- Use Macro F1 for overall ranking
- Inspect per-class F1 to identify weak categories
- Generate confusion matrices and class distributions for qualitative analysis

## Findings

- **Binary performance** is strong with XGBoost and threshold tuning
- **Multi-class performance** benefits from ensemble modelling and per-class threshold adjustment
- **Rare classes** such as Worms and Shellcode remain challenging, even with ADASYN and balanced boosting
- **Cascade design** may improve attack-specific focus, but it must overcome Stage 1 false negatives that prevent attack flows from reaching Stage 2

## Outputs

- `output_png_csv/submission_task1_binary.csv`
- `output_png_csv/submission_task2_multiclass.csv`
- Evaluation plots are saved under `output_png_csv/` with names like:
  - `eval_binary_cm.png`
  - `eval_multi_comparison.png`
  - `eval_perclass_f1.png`
  - `eval_best_multiclass_cm.png`

## How to run

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm catboost jupyter
jupyter notebook k_security_v2.ipynb
```

Set `LOCAL_DATA_DIR` if data is stored outside `./data/`.

## Next steps

- Run Optuna or grid search for hyperparameter tuning on the best flat and cascade models
- Experiment with focal loss or custom class-weighted objectives for rare attack types
- Increase ADASYN sample generation for the smallest classes
- Test a lower cascade threshold to reduce missed attacks in Stage 1
