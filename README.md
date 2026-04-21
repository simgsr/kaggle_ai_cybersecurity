https://github.com/simgsr/kaggle_ai_cybersecurity.git

# Kaggle AI Cybersecurity — Network Intrusion Detection System (v2)

ML-based intrusion detection system trained on the UNSW-NB15 dataset. Solves two tasks: detecting whether network traffic is an attack (binary), and identifying the attack family (multi-class).

## Latest v2 updates

- Hierarchical cascade architecture: binary gate → multi-class only on predicted attacks
- Flat multi-class modelling with LightGBM, CatBoost, and XGBoost
- Stacking ensemble: LGBM + XGB + CatBoost base learners with Logistic Regression meta-learner
- ADASYN oversampling for harder synthetic minority samples
- Per-class threshold tuning to maximise macro F1
- Feature engineering with ratio and log-transformed flow statistics

## Results summary

- Binary classification baseline: **XGBoost** with F1 **0.9303** and ROC-AUC **0.9808**
- Multi-class baseline (v1): **XGBoost Macro F1 0.4818**
- v2 explores stronger multi-class pipelines and ensemble strategies to improve performance on rare attack families

## Dataset

**UNSW-NB15** — network traffic captures with labelled attack categories.

| Split | Records | Features |
|-------|---------|----------|
| Training | 175,341 | 34 |
| Test | 82,332 | 34 |

**Binary target:** `label` — 0 = Normal, 1 = Attack  
**Multi-class target:** `attack_cat` — Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms

## Approach

### Data and preprocessing
- Load training/testing data with CSV/parquet fallback
- Clean dropped unnamed columns and standardise numeric features
- Label encode `proto`, `service`, `state`
- Use ADASYN/SMOTE to balance training data for binary and multi-class tasks
- Apply ratio and log feature engineering on raw flow statistics

### Modelling
- **Task 1 (binary):** XGBoost with threshold tuning and roc/precision/recall evaluation
- **Task 2 (multi-class):** explored both flat and cascade strategies
  - Flat models: XGBoost, LightGBM, CatBoost, Stacking Ensemble
  - Cascade: XGBoost binary gate followed by a dedicated multi-class stage
- Per-class threshold tuning recalibrates class probability cut-offs to maximise macro F1

### Evaluation
- 5-fold Stratified Cross-Validation and held-out test evaluation
- Binary metrics: F1, precision, recall, ROC-AUC
- Multi-class metrics: Macro F1, per-class F1, confusion matrices
- Comparison of flat vs cascade pipelines

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-00A09B)
![CatBoost](https://img.shields.io/badge/CatBoost-Gradient%20Boosting-00B0FF)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

| Library | Purpose |
|---------|---------|
| scikit-learn | preprocessing, CV, metrics, stacking meta-learner |
| XGBoost | binary and multi-class gradient boosting |
| LightGBM | faster gradient boosting baseline and ensemble component |
| CatBoost | imbalanced multi-class modelling with class weights |
| imbalanced-learn | ADASYN / SMOTE oversampling |
| pandas / numpy | data manipulation |
| seaborn / matplotlib | visualisation |

## Project Structure

```
kaggle_ai_cybersecurity/
├── k_security.ipynb        # Original analysis notebook
├── k_security_v2.ipynb     # Improved v2 pipeline and ensemble exploration
├── data/
│   ├── UNSW_NB15_training-set.csv
│   └── UNSW_NB15_testing-set.csv
└── output_png_csv/
    ├── submission_task1_binary.csv
    └── submission_task2_multiclass.csv
```

## Running the Notebook

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm catboost jupyter
jupyter notebook k_security_v2.ipynb
```

> By default the notebook reads data from `./data/`. Set `LOCAL_DATA_DIR` to point to a different directory if needed.

## Key takeaways

- **v2 focuses on better multi-class performance** through ensemble learning, ADASYN oversampling, and per-class threshold tuning.
- **Cascade strategy** tests whether a binary gate can improve multi-class performance by isolating attack flows.
- **Flat stacking and per-class tuning** are designed to recover rare attack classes like Worms and Shellcode more effectively.
- **Submission files** are generated to `output_png_csv/submission_task1_binary.csv` and `output_png_csv/submission_task2_multiclass.csv`.
