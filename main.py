import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, f1_score
from src.nids_model import HierarchicalNIDS

# Constants
DATA_DIR = Path('./data')
MODEL_PATH = Path('./models/nids_hierarchical_v2.skops')
TARGET_BINARY = 'label'
TARGET_MULTI = 'attack_cat'

def load_dataset(data_dir: Path, split: str) -> pd.DataFrame:
    p = data_dir / f'UNSW_NB15_{split}-set.parquet'
    if not p.exists():
        # Fallback to CSV if parquet doesn't exist
        p = data_dir / f'UNSW_NB15_{split}-set.csv'
        return pd.read_csv(p)
    return pd.read_parquet(p)

def main():
    # 1. Load Data
    print("Loading datasets...")
    train_df = load_dataset(DATA_DIR, 'training')
    test_df = load_dataset(DATA_DIR, 'testing')

    # Prepare features and targets
    drop_cols = [TARGET_BINARY, TARGET_MULTI, 'Unnamed: 0', 'id']
    X_train = train_df.drop(columns=drop_cols, errors='ignore')
    y_train_bin = train_df[TARGET_BINARY].values
    y_train_multi = train_df[TARGET_MULTI].values

    X_test = test_df.drop(columns=drop_cols, errors='ignore')
    y_test_multi = test_df[TARGET_MULTI].values

    # 2. Train Model
    nids = HierarchicalNIDS(random_state=42)
    nids.train(X_train, y_train_bin, y_train_multi)

    # 3. Save Model
    nids.save(MODEL_PATH)

    # 4. Evaluate
    print("\nEvaluating on test set...")
    y_pred_idx = nids.predict(X_test)
    y_pred = nids.le_target.inverse_transform(y_pred_idx)
    
    macro_f1 = f1_score(y_test_multi, y_pred, average='macro')
    print(f"\nHierarchical Model Macro F1: {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_multi, y_pred))

    # 5. Verify Loading
    print("\nVerifying model loading...")
    loaded_nids = HierarchicalNIDS.load(MODEL_PATH)
    y_pred_loaded_idx = loaded_nids.predict(X_test.head(10))
    print(f"Loaded model predictions (first 10): {y_pred_loaded_idx}")

if __name__ == "__main__":
    main()
