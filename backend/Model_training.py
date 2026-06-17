"""
HEART DISEASE PREDICTION - MODEL TRAINING MODULE
================================================
Trains and evaluates three machine learning models:
1. Random Forest Classifier (with hyperparameter tuning)
2. Logistic Regression
3. Support Vector Machine (SVM)

Run this ONCE to produce the .pkl model files used by the Streamlit app.
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing helpers from the fixed preprocessing module
from Data_preprocessing import load_data, clean_data, prepare_features, evaluate_model


# ──────────────────────────────────────────────────────────────────────────────
# Thin model wrappers (kept for structural compatibility)
# ──────────────────────────────────────────────────────────────────────────────
class RandomForestModel:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)


class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression(random_state=42, max_iter=1000)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)


class SVMModel:
    def __init__(self):
        # probability=True is required for predict_proba()
        self.model = SVC(probability=True, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)


# ──────────────────────────────────────────────────────────────────────────────
# Main training pipeline
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 STARTING PIPELINE TRAINING ENGINE")
    print("="*60)

    # ── 1. Load → Clean → Prepare ─────────────────────────────────────────────
    # NOTE: load_data() flips the target so 1 = disease, 0 = healthy.
    #       This is the root fix for the HIGH-RISK bug.
    data         = load_data('heart.csv')
    data_cleaned = clean_data(data)

    X_train, X_test, y_train, y_test, scaler, feature_names = \
        prepare_features(data_cleaned)

    # ── 2. Train base estimators ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("🏋️ TRAINING MODELS")
    print("="*60)

    rf_wrapper  = RandomForestModel();  rf_wrapper.train(X_train,  y_train)
    lr_wrapper  = LogisticRegressionModel(); lr_wrapper.train(X_train,  y_train)
    svm_wrapper = SVMModel();           svm_wrapper.train(X_train, y_train)

    print("✅ Random Forest trained")
    print("✅ Logistic Regression trained")
    print("✅ SVM trained")

    # ── 3. Evaluate base models on test set ───────────────────────────────────
    print("\n" + "="*60)
    print("📊 EVALUATING MODELS ON TEST SET")
    print("="*60)

    model_metrics = {}

    for name, wrapper in [
        ('Random Forest',       rf_wrapper),
        ('Logistic Regression', lr_wrapper),
        ('Support Vector Machine', svm_wrapper),
    ]:
        y_pred  = wrapper.model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred, model_name=name)
        model_metrics[name] = metrics

    # ── 4. Hyperparameter tuning for Random Forest ────────────────────────────
    print("\n" + "="*60)
    print("⚙️ TUNING RANDOM FOREST HYPERPARAMETERS")
    print("="*60)

    param_grid = {
        'n_estimators':     [50, 100, 200],
        'max_depth':        [None, 10, 20],
        'min_samples_split': [2, 5],
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    tuned_rf = grid_search.best_estimator_
    print(f"\n   Best params: {grid_search.best_params_}")
    print("   ✅ Hyperparameter optimisation completed!")

    # Evaluate tuned model
    y_pred_tuned = tuned_rf.predict(X_test)
    tuned_metrics = evaluate_model(y_test, y_pred_tuned, model_name="Tuned Random Forest")
    model_metrics['Tuned Random Forest'] = tuned_metrics

    # ── 5. Feature importance from tuned RF ───────────────────────────────────
    feature_importance = dict(
        zip(feature_names, tuned_rf.feature_importances_)
    )

    # ── 6. Save everything ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("💾 SAVING MODELS & ARTIFACTS")
    print("="*60)

    os.makedirs('models', exist_ok=True)

    # Feature name list (needed to align inference input columns)
    with open('models/feature_names.txt', 'w') as f:
        f.write(','.join(list(feature_names)))
    print("✅ Saved: models/feature_names.txt")

    # Models
    # BUG FIX note: we save the inner sklearn estimator (.model), NOT the wrapper,
    # so the Streamlit app's hasattr(model_obj, 'model') check also works either way.
    joblib.dump(tuned_rf,           'models/random_forest_tuned_model.pkl')
    joblib.dump(lr_wrapper.model,   'models/logistic_regression_model.pkl')
    joblib.dump(svm_wrapper.model,  'models/svm_model.pkl')
    joblib.dump(rf_wrapper.model,   'models/random_forest_model.pkl')
    joblib.dump(scaler,             'models/scaler.pkl')
    print("✅ Saved: models/random_forest_tuned_model.pkl")
    print("✅ Saved: models/logistic_regression_model.pkl")
    print("✅ Saved: models/svm_model.pkl")
    print("✅ Saved: models/random_forest_model.pkl")
    print("✅ Saved: models/scaler.pkl")

    # BUG FIX #3 ── Save REAL metrics (not hardcoded) for the analytics tab
    with open('models/model_metrics.json', 'w') as f:
        json.dump(model_metrics, f, indent=2)
    print("✅ Saved: models/model_metrics.json")

    # Save feature importance for analytics tab
    with open('models/feature_importance.json', 'w') as f:
        json.dump(feature_importance, f, indent=2)
    print("✅ Saved: models/feature_importance.json")

    print("\n" + "="*60)
    print("🎉 ALL PIPELINE MODELS EXPORTED SUCCESSFULLY!")
    print("="*60)
