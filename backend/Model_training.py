"""
HEART DISEASE PREDICTION - MODEL TRAINING MODULE
================================================
This script trains and compares three machine learning models:
1. Random Forest Classifier
2. Logistic Regression
3. Support Vector Machine (SVM)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score, 
                             roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


# ========================
# ALGORITHM 1: RANDOM FOREST
# ========================
class RandomForestModel:
    """
    Random Forest Classifier
    
    How it works:
    - Creates multiple decision trees
    - Each tree trained on random subset of data
    - Final prediction = majority vote from all trees
    
    Why good for this project:
    - Handles non-linear relationships
    - Robust to outliers
    - Provides feature importance
    - No feature scaling needed
    """
    
    def __init__(self, n_estimators=100, max_depth=15, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,      # Number of trees
            max_depth=max_depth,            # Maximum depth of each tree
            random_state=random_state,
            n_jobs=-1  # Use all processors
        )
        self.name = "Random Forest Classifier"
        self.feature_importances_ = None
    
    def train(self, X_train, y_train):
        """Train the model"""
        print(f"\n🚀 Training {self.name}...")
        self.model.fit(X_train, y_train)
        self.feature_importances_ = self.model.feature_importances_
        print(f"✅ {self.name} training completed!")
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)


# ========================
# ALGORITHM 2: LOGISTIC REGRESSION
# ========================
class LogisticRegressionModel:
    """
    Logistic Regression
    
    How it works:
    - Uses logistic function to map inputs to probability [0, 1]
    - Finds optimal weights to minimize classification error
    
    Why good for this project:
    - Fast training
    - Interpretable (coefficients show feature importance)
    - Works well for binary classification
    - Low computational cost
    """
    
    def __init__(self, max_iter=1000, random_state=42):
        self.model = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            solver='lbfgs'  # Optimization algorithm
        )
        self.name = "Logistic Regression"
        self.coefficients_ = None
    
    def train(self, X_train, y_train):
        """Train the model"""
        print(f"\n🚀 Training {self.name}...")
        self.model.fit(X_train, y_train)
        self.coefficients_ = self.model.coef_[0]
        print(f"✅ {self.name} training completed!")
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)


# ========================
# ALGORITHM 3: SUPPORT VECTOR MACHINE
# ========================
class SVMModel:
    """
    Support Vector Machine (SVM)
    
    How it works:
    - Finds optimal hyperplane to separate classes
    - Maximizes margin between classes
    - Uses kernel trick for non-linear problems
    
    Why good for this project:
    - Works well with high-dimensional data
    - Effective in cases with clear margin of separation
    - Memory efficient
    """
    
    def __init__(self, kernel='rbf', C=1.0, random_state=42):
        self.model = SVC(
            kernel=kernel,          # 'rbf' = Radial Basis Function
            C=C,                    # Regularization parameter
            random_state=random_state,
            probability=True        # Enable probability estimates
        )
        self.name = "Support Vector Machine (SVM)"
    
    def train(self, X_train, y_train):
        """Train the model"""
        print(f"\n🚀 Training {self.name}...")
        self.model.fit(X_train, y_train)
        print(f"✅ {self.name} training completed!")
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)


# ========================
# EVALUATION FUNCTION
# ========================
def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model on test set
    
    Args:
        model: Trained model object
        X_test: Test features
        y_test: Test target
        model_name: Name for display
    
    Returns:
        dict: Performance metrics
    """
    print(f"\n{'='*60}")
    print(f"📊 {model_name} - TEST SET EVALUATION")
    print(f"{'='*60}")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n✅ ACCURACY:   {accuracy*100:.2f}%")
    print(f"✅ PRECISION:  {precision*100:.2f}%")
    print(f"✅ RECALL:     {recall*100:.2f}%")
    print(f"✅ F1-SCORE:   {f1:.4f}")
    print(f"✅ ROC-AUC:    {roc_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n📈 CONFUSION MATRIX:")
    print(f"   True Negatives:   {tn}")
    print(f"   False Positives:  {fp}")
    print(f"   False Negatives:  {fn}")
    print(f"   True Positives:   {tp}")
    
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    
    print(f"\n   Sensitivity (Recall):  {sensitivity*100:.2f}%")
    print(f"   Specificity:           {specificity*100:.2f}%")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return metrics


# ========================
# HYPERPARAMETER TUNING
# ========================
def tune_random_forest(X_train, y_train, X_test, y_test):
    """
    Use GridSearchCV to find best hyperparameters for Random Forest
    
    Hyperparameters to tune:
    - n_estimators: Number of trees (more = better but slower)
    - max_depth: Maximum depth (deeper = more complex)
    - min_samples_split: Min samples to split node (higher = less overfitting)
    """
    print(f"\n{'='*60}")
    print("🔧 HYPERPARAMETER TUNING - Random Forest")
    print(f"{'='*60}")
    
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    
    print("\nTesting parameter combinations...")
    print("This may take a moment...\n")
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"✅ Best parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    print(f"\n   Best F1-Score (CV): {grid_search.best_score_:.4f}")
    
    # Train model with best parameters
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Test Set Accuracy: {accuracy*100:.2f}%")
    
    return best_model


# ========================
# COMPARISON VISUALIZATION
# ========================
def compare_models(models_results, feature_names=None):
    """
    Create visualizations comparing all models
    
    Args:
        models_results: Dict with model names and their metrics
        feature_names: Names of features (for feature importance)
    """
    print(f"\n{'='*60}")
    print("📊 MODEL COMPARISON & VISUALIZATION")
    print(f"{'='*60}")
    
    # 1. Metrics Comparison Bar Chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Heart Disease Prediction - Model Comparison', fontsize=16, fontweight='bold')
    
    metrics_list = ['accuracy', 'precision', 'recall', 'f1_score']
    colors = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green
    
    for idx, metric in enumerate(metrics_list):
        ax = axes[idx // 2, idx % 2]
        
        values = [models_results[model][metric] for model in models_results.keys()]
        model_names = list(models_results.keys())
        
        bars = ax.bar(model_names, values, color=colors[:len(model_names)], alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height*100:.2f}%',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_title(metric.upper().replace('_', ' '), fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: model_comparison.png")
    plt.show()


# ========================
# MAIN TRAINING PIPELINE
# ========================
def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("🏥 HEART DISEASE PREDICTION - MODEL TRAINING & COMPARISON")
    print("="*70)
    
    # Load and prepare data
    print("\n📥 Loading and preparing data...")
    from Data_preprocessing import load_data, clean_data, prepare_features
    
    # Load data
    data = load_data('heart_disease_data.csv')
    data_clean = clean_data(data)
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_features(data_clean)
    
    # ========================
    # TRAIN ALL THREE MODELS
    # ========================
    print(f"\n{'='*70}")
    print("🚀 TRAINING ALL MODELS")
    print(f"{'='*70}")
    
    # Model 1: Random Forest
    rf_model = RandomForestModel(n_estimators=100, max_depth=15)
    rf_model.train(X_train, y_train)
    rf_metrics = evaluate_model(rf_model.model, X_test, y_test, "Random Forest Classifier")
    
    # Model 2: Logistic Regression
    lr_model = LogisticRegressionModel()
    lr_model.train(X_train, y_train)
    lr_metrics = evaluate_model(lr_model.model, X_test, y_test, "Logistic Regression")
    
    # Model 3: SVM
    svm_model = SVMModel(kernel='rbf', C=1.0)
    svm_model.train(X_train, y_train)
    svm_metrics = evaluate_model(svm_model.model, X_test, y_test, "Support Vector Machine")
    
    # ========================
    # HYPERPARAMETER TUNING (Optional)
    # ========================
    print(f"\n{'='*70}")
    print("🔧 HYPERPARAMETER TUNING")
    print(f"{'='*70}")
    
    tuned_rf = tune_random_forest(X_train, y_train, X_test, y_test)
    tuned_rf_metrics = evaluate_model(tuned_rf, X_test, y_test, "Random Forest (Tuned)")
    
    # ========================
    # MODEL COMPARISON
    # ========================
    models_comparison = {
        'Random Forest': rf_metrics,
        'Logistic Regression': lr_metrics,
        'Support Vector Machine': svm_metrics,
        'Random Forest (Tuned)': tuned_rf_metrics
    }
    
    # Determine best model
    best_model_name = max(models_comparison.keys(), 
                         key=lambda x: models_comparison[x]['f1_score'])
    
    print(f"\n{'='*70}")
    print("🏆 MODEL COMPARISON RESULTS")
    print(f"{'='*70}")
    
    print(f"\n{'Model':<30} {'Accuracy':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
    print("-" * 66)
    for model_name, metrics in models_comparison.items():
        print(f"{model_name:<30} {metrics['accuracy']*100:>9.2f}% {metrics['f1_score']:>11.4f} {metrics['roc_auc']:>11.4f}")
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   F1-Score: {models_comparison[best_model_name]['f1_score']:.4f}")
    
    # ========================
    # SAVE MODELS
    # ========================
    import os
    os.makedirs('models', exist_ok=True)   

    joblib.dump(rf_model.model, 'models/random_forest_model.pkl')
    joblib.dump(lr_model.model, 'models/logistic_regression_model.pkl')
    print(f"\n{'='*70}")
    print("💾 SAVING MODELS")
    print(f"{'='*70}")
    
    joblib.dump(rf_model.model, 'models/random_forest_model.pkl')
    print("✅ Saved: random_forest_model.pkl")
    
    joblib.dump(lr_model.model, 'models/logistic_regression_model.pkl')
    print("✅ Saved: logistic_regression_model.pkl")
    
    joblib.dump(svm_model.model, 'models/svm_model.pkl')
    print("✅ Saved: svm_model.pkl")
    
    joblib.dump(tuned_rf, 'models/random_forest_tuned_model.pkl')
    print("✅ Saved: random_forest_tuned_model.pkl")
    
    joblib.dump(scaler, 'models/scaler.pkl')
    print("✅ Saved: scaler.pkl")
    
    # Save feature names
    with open('models/feature_names.txt', 'w') as f:
        f.write(','.join(feature_names))
    print("✅ Saved: feature_names.txt")
    
    # Visualize comparison
    compare_models(models_comparison, feature_names)
    
    print(f"\n{'='*70}")
    print("✅ MODEL TRAINING COMPLETED!")
    print(f"{'='*70}")
    print(f"\nNext steps:")
    print("1. Use Streamlit app for web interface: streamlit run streamlit_app.py")
    print("2. Deploy to cloud platform")
    print("3. Monitor model performance in production")


if __name__ == "__main__":
    main()
