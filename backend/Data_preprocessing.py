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
    Load heart disease dataset from CSV file
    
    Args:
        filepath (str): Path to the CSV file
    
    Returns:
        DataFrame: Loaded data
    """
    print("📁 Loading data...")
    data = pd.read_csv(filepath)
    print(f"✅ Data loaded successfully!")
    print(f"   Dataset Shape: {data.shape[0]} rows, {data.shape[1]} columns")
    return data


# ========================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ========================
def explore_data(data):
    """
    Analyze and visualize the dataset
    
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
    print(f"   No Disease (0): {target_counts[0]} patients ({target_counts[0]/len(data)*100:.1f}%)")
    print(f"   Disease (1):    {target_counts[1]} patients ({target_counts[1]/len(data)*100:.1f}%)")
    
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
    Clean data by handling missing values and outliers
    
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
        # Fill numeric columns with median
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
    
    # Handle outliers using IQR method (optional - be careful)
    print("\n📍 Outlier Detection (IQR Method):")
    numeric_cols = data_clean.select_dtypes(include=[np.number]).columns
    outlier_count = 0
    
    for col in numeric_cols:
        Q1 = data_clean[col].quantile(0.25)
        Q3 = data_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_in_col = len(data_clean[(data_clean[col] < lower_bound) | (data_clean[col] > upper_bound)])
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
    Separate features and target, scale features, split into train-test
    
    Args:
        data (DataFrame): Cleaned dataset
        target_col (str): Name of target column
        test_size (float): Proportion of data for testing (0-1)
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    print("\n" + "="*50)
    print("⚙️ FEATURE PREPARATION & SCALING")
    print("="*50)
    
    # Separate features (X) and target (y)
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    print(f"\n1️⃣ FEATURES & TARGET SEPARATION:")
    print(f"   Features (X): {X.shape}")
    print(f"   Target (y): {y.shape}")
    print(f"   Feature names: {list(X.columns)}")
    
    # Train-test split
    print(f"\n2️⃣ TRAIN-TEST SPLIT ({(1-test_size)*100:.0f}%-{test_size*100:.0f}%):")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Maintains class distribution
    )
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Testing set: {X_test.shape[0]} samples")
    
    # Feature scaling (VERY IMPORTANT for many algorithms)
    print(f"\n3️⃣ FEATURE SCALING (StandardScaler):")
    print("   StandardScaler formula: (X - mean) / std_dev")
    print("   This brings all features to similar scale (mean=0, std=1)")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for readability
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    print("   ✅ Scaling completed!")
    print(f"   Scaled features shape: {X_train_scaled.shape}")
    
    # Show before and after scaling
    print("\n4️⃣ BEFORE AND AFTER SCALING:")
    print("   Original feature ranges:")
    print(f"      Age: {X_train['age'].min():.1f} to {X_train['age'].max():.1f}")
    print(f"      Chol: {X_train['chol'].min():.1f} to {X_train['chol'].max():.1f}")
    print("\n   Scaled feature ranges:")
    print(f"      Age: {X_train_scaled['age'].min():.2f} to {X_train_scaled['age'].max():.2f}")
    print(f"      Chol: {X_train_scaled['chol'].min():.2f} to {X_train_scaled['chol'].max():.2f}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns


# ========================
# STEP 5: EVALUATE MODELS
# ========================
def evaluate_model(y_true, y_pred, y_pred_proba=None, model_name="Model"):
    """
    Evaluate model performance with multiple metrics
    
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
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"✅ Accuracy:  {accuracy*100:.2f}%  (Correct predictions)")
    print(f"✅ Precision: {precision*100:.2f}%  (Positive predictive value)")
    print(f"✅ Recall:    {recall*100:.2f}%   (True positive rate)")
    print(f"✅ F1-Score:  {f1:.4f}      (Harmonic mean)")
    
    # Confusion matrix details
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n📊 CONFUSION MATRIX:")
    print(f"   True Negatives (Correct No Disease):  {tn}")
    print(f"   False Positives (Incorrectly Pred Disease):  {fp}")
    print(f"   False Negatives (Missed Disease Cases):  {fn}")
    print(f"   True Positives (Correct Disease):  {tp}")
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\n📈 ADDITIONAL METRICS:")
    print(f"   Sensitivity (Recall):    {sensitivity*100:.2f}%")
    print(f"   Specificity:             {specificity*100:.2f}%")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'sensitivity': sensitivity,
        'specificity': specificity
    }
    
    return metrics


# ========================
# MAIN EXECUTION
# ========================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🏥 HEART DISEASE PREDICTION - DATA PREPROCESSING")
    print("="*60)
    
    # Step 1: Load data
    data = load_data('heart_disease_data.csv')
    
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
    print("Use model_training.py to train models")