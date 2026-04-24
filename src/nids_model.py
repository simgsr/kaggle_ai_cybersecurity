import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import skops.io as sio

class HierarchicalNIDS:
    """
    Hierarchical Network Intrusion Detection System (NIDS).
    Stage 1: Binary classification (Normal vs. Attack).
    Stage 2: Multi-class classification (Attack Category) for predicted attacks.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.le_target = LabelEncoder()
        self.le_threats = LabelEncoder()
        self.cat_encoders = {}
        self.model_s1 = None
        self.model_s2 = None
        self.categorical_cols = ['proto', 'service', 'state']
        self.normal_idx = None

    def preprocess_train(self, X, y_bin, y_multi):
        """Preprocess training data and fit encoders."""
        X_processed = X.copy()
        
        # 1. Encode Categoricals
        for col in self.categorical_cols:
            le = LabelEncoder()
            # Convert to string and fit
            data = X_processed[col].astype(str)
            le.fit(data)
            
            # Add 'Unknown' to classes if not present
            le.classes_ = np.append(le.classes_, 'Unknown')
            
            X_processed[col] = le.transform(data)
            self.cat_encoders[col] = le
            
        # 2. Scale Features
        X_scaled = self.scaler.fit_transform(X_processed)
        
        # 3. Fit Target Encoders
        self.le_target.fit(y_multi)
        self.normal_idx = self.le_target.transform(['Normal'])[0]
        
        return X_scaled

    def preprocess_inference(self, X):
        """Preprocess data for inference using fitted encoders."""
        X_processed = X.copy()
        for col in self.categorical_cols:
            le = self.cat_encoders[col]
            data = X_processed[col].astype(str)
            
            # Map unseen labels to 'Unknown'
            # Note: le.classes_ contains 'Unknown' at the last index
            known_labels = set(le.classes_)
            data = data.map(lambda x: x if x in known_labels else 'Unknown')
            
            X_processed[col] = le.transform(data)
            
        X_scaled = self.scaler.transform(X_processed)
        return X_scaled

    def train(self, X_train, y_train_bin, y_train_multi):
        """Train the hierarchical model."""
        print("Preprocessing training data...")
        X_train_scaled = self.preprocess_train(X_train, y_train_bin, y_train_multi)
        
        # Stage 1: Binary (Normal vs Threat)
        print("Training Stage 1 (Binary)...")
        smote_bin = SMOTE(random_state=self.random_state)
        X_sm_b, y_sm_b = smote_bin.fit_resample(X_train_scaled, y_train_bin)
        self.model_s1 = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, 
                                      random_state=self.random_state, n_jobs=-1)
        self.model_s1.fit(X_sm_b, y_sm_b)
        
        # Stage 2: Multi-class (Threats Only)
        print("Training Stage 2 (Multi-class)...")
        # Get target indices for multi-class
        y_train_multi_idx = self.le_target.transform(y_train_multi)
        
        mask_threat = (y_train_bin == 1)
        X_threats = X_train_scaled[mask_threat]
        y_threats = y_train_multi_idx[mask_threat]
        
        # Re-encode threats to 0-N for XGBoost
        y_threats_enc = self.le_threats.fit_transform(y_threats)
        
        smote_threats = SMOTE(random_state=self.random_state)
        X_threats_sm, y_threats_sm = smote_threats.fit_resample(X_threats, y_threats_enc)
        
        self.model_s2 = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, 
                                      random_state=self.random_state, n_jobs=-1)
        self.model_s2.fit(X_threats_sm, y_threats_sm)
        print("Training completed.")

    def predict(self, X):
        """Perform hierarchical inference."""
        X_scaled = self.preprocess_inference(X)
        
        # Predict Stage 1
        y_pred_s1 = self.model_s1.predict(X_scaled)
        y_pred_hierarchical = np.full(len(y_pred_s1), self.normal_idx)
        
        # Predict Stage 2 for threats
        threat_mask = (y_pred_s1 == 1)
        if threat_mask.any():
            threat_X = X_scaled[threat_mask]
            threat_preds_enc = self.model_s2.predict(threat_X)
            # Inverse transform from Stage 2 encoding to multi-class index
            threat_preds_idx = self.le_threats.inverse_transform(threat_preds_enc)
            y_pred_hierarchical[threat_mask] = threat_preds_idx
            
        return y_pred_hierarchical

    def save(self, file_path):
        """Save the model securely using skops."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        # Store all components in a dictionary
        state = {
            "random_state": self.random_state,
            "scaler": self.scaler,
            "le_target": self.le_target,
            "le_threats": self.le_threats,
            "cat_encoders": self.cat_encoders,
            "model_s1": self.model_s1,
            "model_s2": self.model_s2,
            "categorical_cols": self.categorical_cols,
            "normal_idx": self.normal_idx
        }
        sio.dump(state, file_path)
        print(f"Model saved to {file_path} using skops.")

    @classmethod
    def load(cls, file_path):
        """Load the model securely using skops."""
        # For maximum security, we retrieve the types present in the file
        # and pass them as trusted after verification.
        trusted_types = sio.get_untrusted_types(file=file_path)
        state = sio.load(file_path, trusted=trusted_types) 
        instance = cls(random_state=state["random_state"])
        instance.scaler = state["scaler"]
        instance.le_target = state["le_target"]
        instance.le_threats = state["le_threats"]
        instance.cat_encoders = state["cat_encoders"]
        instance.model_s1 = state["model_s1"]
        instance.model_s2 = state["model_s2"]
        instance.categorical_cols = state["categorical_cols"]
        instance.normal_idx = state["normal_idx"]
        return instance
