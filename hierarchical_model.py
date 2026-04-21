import os, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# Constants
RANDOM_STATE = 42
TARGET_BINARY = 'label'
TARGET_MULTI = 'attack_cat'
CATEGORICAL_COLS = ['proto', 'service', 'state']

def load_dataset(data_dir: Path, split: str) -> pd.DataFrame:
    p = data_dir / f'UNSW_NB15_{split}-set.parquet'
    return pd.read_parquet(p)

DATA_DIR = Path('./data')
train_df = load_dataset(DATA_DIR, 'training')
test_df = load_dataset(DATA_DIR, 'testing')

# 1. Preprocessing
X_train_raw = train_df.drop(columns=[TARGET_BINARY, TARGET_MULTI, 'Unnamed: 0'], errors='ignore')
X_test_raw = test_df.drop(columns=[TARGET_BINARY, TARGET_MULTI, 'Unnamed: 0'], errors='ignore')

y_train_bin = train_df[TARGET_BINARY].values
y_test_bin = test_df[TARGET_BINARY].values
y_train_cat = train_df[TARGET_MULTI].values
y_test_cat = test_df[TARGET_MULTI].values

# Encode categoricals
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    combined = pd.concat([X_train_raw[col], X_test_raw[col]]).astype(str)
    le.fit(combined)
    X_train_raw[col] = le.transform(X_train_raw[col].astype(str))
    X_test_raw[col] = le.transform(X_test_raw[col].astype(str))

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Encode multi-class target
le_target = LabelEncoder()
le_target.fit(pd.concat([train_df[TARGET_MULTI], test_df[TARGET_MULTI]]))
y_train_multi = le_target.transform(y_train_cat)
y_test_multi = le_target.transform(y_test_cat)

# 2. Train Single Multi-class Model (Baseline)
print("Training Baseline Multi-class Model...")
smote_multi = SMOTE(random_state=RANDOM_STATE)
X_train_sm_m, y_train_sm_m = smote_multi.fit_resample(X_train_scaled, y_train_multi)

model_baseline = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE, n_jobs=-1)
model_baseline.fit(X_train_sm_m, y_train_sm_m)
y_pred_baseline = model_baseline.predict(X_test_scaled)
baseline_f1 = f1_score(y_test_multi, y_pred_baseline, average='macro')
print(f"Baseline Macro F1: {baseline_f1:.4f}")

# 3. Hierarchical Model
print("\nTraining Hierarchical Model...")

# Stage 1: Binary (Normal vs Threat)
smote_bin = SMOTE(random_state=RANDOM_STATE)
X_train_sm_b, y_train_sm_b = smote_bin.fit_resample(X_train_scaled, y_train_bin)
model_s1 = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE, n_jobs=-1)
model_s1.fit(X_train_sm_b, y_train_sm_b)

# Stage 2: Threats only (9 classes)
mask_threat = (y_train_bin == 1)
X_train_threats = X_train_scaled[mask_threat]
y_train_threats = y_train_multi[mask_threat]

# Re-encode threats to 0-8 for XGBoost
le_threats = LabelEncoder()
y_train_threats_enc = le_threats.fit_transform(y_train_threats)

smote_threats = SMOTE(random_state=RANDOM_STATE)
X_train_threats_sm, y_train_threats_sm = smote_threats.fit_resample(X_train_threats, y_train_threats_enc)

model_s2 = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE, n_jobs=-1)
model_s2.fit(X_train_threats_sm, y_train_threats_sm)

# Inference
y_pred_s1 = model_s1.predict(X_test_scaled)
y_pred_hierarchical = np.zeros_like(y_pred_s1)

# Normal class index
normal_idx = le_target.transform(['Normal'])[0]

# For each prediction
for i in range(len(y_pred_s1)):
    if y_pred_s1[i] == 0:
        y_pred_hierarchical[i] = normal_idx
    else:
        # Predict threat category
        threat_pred_enc = model_s2.predict(X_test_scaled[i:i+1])[0]
        y_pred_hierarchical[i] = le_threats.inverse_transform([threat_pred_enc])[0]

hierarchical_f1 = f1_score(y_test_multi, y_pred_hierarchical, average='macro')
print(f"Hierarchical Macro F1: {hierarchical_f1:.4f}")

# Comparison Report
if hierarchical_f1 > baseline_f1:
    print("\n✅ Hierarchical model performed BETTER.")
else:
    print("\n❌ Hierarchical model performed WORSE.")

