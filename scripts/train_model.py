#!/usr/bin/env python3
"""
ML MODEL TRAINING SCRIPT
Trains a Random Forest classifier on the extracted features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

print("="*60)
print("🛡️  ML-BASED NIDS - MODEL TRAINING")
print("="*60)

# 1. LOAD DATA
datasets_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')
csv_path = os.path.join(datasets_dir, 'network_traffic_dataset.csv')
print(f"\n📥 Loading dataset from {csv_path}...")
df = pd.read_csv(csv_path)
print(f"   ✅ Loaded {len(df)} records")

# 2. PREPARE FEATURES (X) AND LABELS (y)
print("\n🔧 Preparing features...")

feature_cols = [col for col in df.columns if col not in ['label', 'src_port']]
X = df[feature_cols]
y = df['label']

# Encode labels (convert 'ssh_bruteforce' → 0, 'normal' → 1, etc.)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"   Features: {list(feature_cols)}")
print(f"   Classes: {list(le.classes_)}")
print(f"   Class mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 3. SPLIT DATA (70% train, 30% test)
print("\n✂️  Splitting data (70% train / 30% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)
print(f"   Train size: {len(X_train)} | Test size: {len(X_test)}")

# 4. TRAIN MODEL
print("\n🧠 Training Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("   ✅ Training complete!")

# 5. EVALUATE
print("\n📊 EVALUATION RESULTS")
print("="*60)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 ACCURACY: {accuracy*100:.2f}%")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\n🔢 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature Importance
print("\n⭐ TOP 5 IMPORTANT FEATURES:")
importances = model.feature_importances_
feature_imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': importances
}).sort_values('importance', ascending=False)
print(feature_imp.head(5).to_string(index=False))

# 6. SAVE MODEL
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, 'random_forest_nids.pkl')
encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
joblib.dump(model, model_path)
joblib.dump(le, encoder_path)
print(f"\n💾 Model saved to {model_path}")
print(f"   Encoder saved to {encoder_path}")

print("\n" + "="*60)
print("✅ MODEL TRAINING COMPLETE!")
print("="*60)