import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# --- CONFIGURATION ---
TRAIN_PATH = 'train.csv'
TEST_PATH = 'test.csv'
MODEL_SAVE_PATH = 'stress_model_rf.pkl'

def load_and_preprocess():
    """Loads data, encodes the target, and splits into features/labels."""
    print("⏳ Loading data...")
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        print(f"❌ Error: '{TRAIN_PATH}' or '{TEST_PATH}' not found. Please ensure they are in the same folder.")
        return None, None, None, None, None

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Encode Target Variable (0=Interruption, 1=No Stress, 2=Time Pressure)
    le = LabelEncoder()
    train_df['condition_encoded'] = le.fit_transform(train_df['condition'])
    test_df['condition_encoded'] = le.transform(test_df['condition'])
    
    # Save the label mapping for reference
    labels = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"✅ Data Loaded. Classes: {labels}")

    # Separate Features and Target
    # Dropping non-feature columns
    X_train = train_df.drop(['datasetId', 'condition', 'condition_encoded'], axis=1)
    y_train = train_df['condition_encoded']

    X_test = test_df.drop(['datasetId', 'condition', 'condition_encoded'], axis=1)
    y_test = test_df['condition_encoded']

    return X_train, y_train, X_test, y_test, le

def train_model(X_train, y_train):
    """Trains a Random Forest Classifier."""
    print("⏳ Training Random Forest Model (this may take a moment)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("✅ Model Trained successfully!")
    return model

def evaluate_model(model, X_test, y_test, classes):
    """Prints accuracy and classification report."""
    print("⏳ Evaluating model...")
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🏆 Test Set Accuracy: {acc*100:.2f}%")

    # Classification Report
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=classes))

    return y_pred

if __name__ == "__main__":
    # 1. Load Data
    X_train, y_train, X_test, y_test, le = load_and_preprocess()
    
    if X_train is not None:
        # 2. Train
        rf_model = train_model(X_train, y_train)

        # 3. Save Model
        joblib.dump(rf_model, MODEL_SAVE_PATH)
        print(f"💾 Model saved to {MODEL_SAVE_PATH}")

        # 4. Evaluate
        class_names = [str(c) for c in le.classes_]
        y_pred = evaluate_model(rf_model, X_test, y_test, class_names)
        
        print("\n🎉 Process Complete!")
