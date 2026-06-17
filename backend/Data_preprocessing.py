"""
HEART DISEASE PREDICTION - DATA PREPROCESSING MODULE
=====================================
This script handles:
1. Data loading
2. Exploratory Data Analysis (EDA)
3. Data cleaning & handling missing values
4. Feature scaling/normalization
5. Train-test split
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ========================
# STEP 1: LOAD DATA
# ========================
def load_data(filepath):
    """
    Load heart disease dataset from CSV file.

    Args:
        filepath (str): Path to the CSV file

    Returns:
        DataFrame: Loaded data
    """
    print("📁 Loading data...")
    data = pd.read_csv(filepath)
    print(f"✅ Data loaded successfully!")
    print(f"   Dataset Shape: {data.shape[0]} rows, {data.shape[1]} columns")

    # ─────────────────────────────────────────────────────────────────
    # BUG FIX #1 ── TARGET COLUMN IS INVERTED IN THIS DATASET
    # In this Kaggle version: 0 = heart disease, 1 = no heart disease
    # We flip it so the standard convention holds: 1 = disease, 0 = healthy
    # Evidence: patients with target=1 have HIGHER thalach (healthy sign),
    #           LOWER exang, LOWER ca, LOWER oldpeak – all healthy markers.
    # Without this fix every prediction is backwards and HIGH RISK is
    # impossible to reach.
    # ─────────────────────────────────────────────────────────────────
    data['target'] = 1 - data['target']
    print("   ✅ Target variable corrected: 1 = disease, 0 = no disease")

    return data


# ========================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ========================
def explore_data(data):
    """
    Analyse and visualise the dataset.

    Args:
        data (DataFrame): The dataset to explore
    """
    print("\n" + "="*50)
    print("📊 DATA EXPLORATION")
    print("="*50)

    # Display basic info
    print("\n1️⃣ DATASET INFO:")
    print(f"   Shape: {data.shape}")
    print(f"   Data Types:\n{data.dtypes}")

    # Missing values
    print("\n2️⃣ MISSING VALUES:")
    missing = data.isnull().sum()
    if missing.sum() == 0:
        print("   ✅ No missing values found!")
    else:
        print(missing[missing > 0])

    # Statistical summary
    print("\n3️⃣ STATISTICAL SUMMARY:")
    print(data.describe().round(2))

    # Target distribution
    print("\n4️⃣ TARGET VARIABLE DISTRIBUTION:")
    target_counts = data['target'].value_counts()
    print(f"   No Disease (0): {target_counts.get(0, 0)} patients "
          f"({target_counts.get(0, 0)/len(data)*100:.1f}%)")
    print(f"   Disease   (1): {target_counts.get(1, 0)} patients "
          f"({target_counts.get(1, 0)/len(data)*100:.1f}%)")

    # Feature correlations with target
    print("\n5️⃣ FEATURE CORRELATION WITH TARGET:")
    correlations = data.corr()['target'].sort_values(ascending=False)
    print(correlations)

    return correlations


# ========================
# STEP 3: DATA CLEANING
# ========================
def clean_data(data):
    """
    Clean data by handling missing values and outliers.

    Args:
        data (DataFrame): Raw dataset

    Returns:
        DataFrame: Cleaned dataset
    """
    print("\n" + "="*50)
    print("🧹 DATA CLEANING")
    print("="*50)

    # Create a copy to avoid modifying original
    data_clean = data.copy()

    # Handle missing values (if any)
    if data_clean.isnull().sum().sum() > 0:
        print("\n Handling missing values...")
        numeric_columns = data_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            data_clean[col].fillna(data_clean[col].median(), inplace=True)
        print("✅ Missing values handled!")

    # Remove duplicates (if any)
    initial_rows = len(data_clean)
    data_clean.drop_duplicates(inplace=True)
    removed = initial_rows - len(data_clean)
    if removed > 0:
        print(f"\n🔄 Removed {removed} duplicate rows")
    else:
        print("\n✅ No duplicates found")

    # Outlier detection using IQR method
    print("\n📍 Outlier Detection (IQR Method):")
    numeric_cols = data_clean.select_dtypes(include=[np.number]).columns
    outlier_count = 0

    for col in numeric_cols:
        Q1 = data_clean[col].quantile(0.25)
        Q3 = data_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_in_col = len(
            data_clean[(data_clean[col] < lower_bound) | (data_clean[col] > upper_bound)]
        )
        if outliers_in_col > 0:
            outlier_count += outliers_in_col
            print(f"   {col}: {outliers_in_col} outliers detected")

    if outlier_count == 0:
        print("   ✅ No significant outliers found")

    print(f"\n✅ Data cleaning completed!")
    print(f"   Final dataset shape: {data_clean.shape}")

    return data_clean


# ========================
# STEP 4: FEATURE ENGINEERING & SCALING
# ========================
def prepare_features(data, target_col='target', test_size=0.2, random_state=42):
    """
    Encode, split, and scale features.

    Args:
        data (DataFrame): Cleaned dataset
        target_col (str): Name of the target column
        test_size (float): Fraction reserved for testing
        random_state (int): Reproducibility seed

    Returns:
        Tuple of X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns
    """
    print("\n" + "="*50)
    print("⚙️ FEATURE PREPARATION & SCALING")
    print("="*50)

    # Separate features (X) and target (y)
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # One-hot encode multi-class categorical columns
    # BUG FIX #2 ── 'thal' has values 0,1,2,3 in the data.
    # The original code's Streamlit selectbox only offered [1,2,3],
    # meaning thal_0 was never produced at inference, causing silent
    # feature-alignment errors. We encode all values present in the
    # training data; the app now also sends thal_0 when needed.
    categorical_cols = ['cp', 'restecg', 'slope', 'thal']
    for col in categorical_cols:
        X[col] = X[col].astype(str)

    X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

    print(f"\n1️⃣ FEATURES & TARGET SEPARATION:")
    print(f"   Features (X): {X.shape}")
    print(f"   Target (y):   {y.shape}")
    print(f"   Feature columns: {list(X.columns)}")

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Scale only continuous numeric columns; leave binary dummy columns untouched
    continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled  = X_test.copy()

    X_train_scaled[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test_scaled[continuous_cols]  = scaler.transform(X_test[continuous_cols])

    print(f"\n2️⃣ TRAIN / TEST SPLIT:")
    print(f"   Training samples: {X_train_scaled.shape[0]}")
    print(f"   Testing samples:  {X_test_scaled.shape[0]}")
    print("   ✅ Encoding and scaling completed!")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns


# ========================
# STEP 5: EVALUATE MODELS
# ========================
def evaluate_model(y_true, y_pred, y_pred_proba=None, model_name="Model"):
    """
    Evaluate model performance with multiple metrics.

    Args:
        y_true: Actual target values
        y_pred: Predicted values
        y_pred_proba: Prediction probabilities (optional)
        model_name: Name of the model for display

    Returns:
        dict: Dictionary with all evaluation metrics
    """
    print(f"\n{model_name} EVALUATION:")
    print("="*50)

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred)

    print(f"✅ Accuracy:  {accuracy*100:.2f}%  (Correct predictions)")
    print(f"✅ Precision: {precision*100:.2f}%  (Positive predictive value)")
    print(f"✅ Recall:    {recall*100:.2f}%   (True positive rate)")
    print(f"✅ F1-Score:  {f1:.4f}      (Harmonic mean)")

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n📊 CONFUSION MATRIX:")
    print(f"   True Negatives  (Correct No Disease):        {tn}")
    print(f"   False Positives (Incorrectly Pred Disease):  {fp}")
    print(f"   False Negatives (Missed Disease Cases):      {fn}")
    print(f"   True Positives  (Correct Disease):           {tp}")

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"\n📈 ADDITIONAL METRICS:")
    print(f"   Sensitivity (Recall): {sensitivity*100:.2f}%")
    print(f"   Specificity:          {specificity*100:.2f}%")

    metrics = {
        'accuracy':    accuracy,
        'precision':   precision,
        'recall':      recall,
        'f1_score':    f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
    }
    return metrics


# ========================
# MAIN EXECUTION
# ========================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🏥 HEART DISEASE PREDICTION - DATA PREPROCESSING")
    print("="*60)

    # Step 1: Load data (target flip applied inside load_data)
    data = load_data('heart.csv')

    # Step 2: Explore data
    correlations = explore_data(data)

    # Step 3: Clean data
    data_clean = clean_data(data)

    # Step 4: Prepare features
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_features(data_clean)

    print("\n" + "="*60)
    print("✅ DATA PREPROCESSING COMPLETED!")
    print("="*60)
    print("\nNext Step: Train machine learning models on this data")
    print("Run  model_training.py  to train and save models.")