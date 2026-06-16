"""
HEART DISEASE PREDICTION - FLASK WEB APPLICATION (Alternative to Streamlit)
===========================================================================
Flask app for heart disease prediction with REST API and HTML interface
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# ========================
# LOAD MODELS & SCALER
# ========================
try:
    rf_model = joblib.load('models/random_forest_tuned_model.pkl')
    lr_model = joblib.load('models/logistic_regression_model.pkl')
    svm_model = joblib.load('models/svm_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    with open('models/feature_names.txt', 'r') as f:
        feature_names = f.read().strip().split(',')
    
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("Please train models first using model_training.py")

# ========================
# HELPER FUNCTIONS
# ========================
def categorize_risk(probability):
    """Categorize risk level based on probability"""
    if probability < 0.3:
        return {
            'level': 'LOW_RISK',
            'percentage': probability * 100,
            'message': 'Low probability of heart disease. Continue healthy lifestyle.',
            'color': '#28a745'
        }
    elif probability < 0.7:
        return {
            'level': 'MEDIUM_RISK',
            'percentage': probability * 100,
            'message': 'Moderate risk. Please consult a cardiologist.',
            'color': '#ffc107'
        }
    else:
        return {
            'level': 'HIGH_RISK',
            'percentage': probability * 100,
            'message': 'High probability. Seek immediate medical attention!',
            'color': '#dc3545'
        }


def predict_disease(patient_data):
    """
    Make prediction for a patient
    
    Args:
        patient_data: Dictionary with patient features
    
    Returns:
        dict: Predictions from all models and risk assessment
    """
    # Create DataFrame
    input_df = pd.DataFrame([patient_data])
    
    # Ensure correct column order
    input_df = input_df[feature_names]
    
    # Scale features
    input_scaled = scaler.transform(input_df)
    
    # Predictions from all models
    rf_pred = rf_model.predict(input_scaled)[0]
    rf_proba = rf_model.predict_proba(input_scaled)[0][1]
    
    lr_pred = lr_model.predict(input_scaled)[0]
    lr_proba = lr_model.predict_proba(input_scaled)[0][1]
    
    svm_pred = svm_model.predict(input_scaled)[0]
    svm_proba = svm_model.predict_proba(input_scaled)[0][1]
    
    # Average probability
    avg_probability = np.mean([rf_proba, lr_proba, svm_proba])
    
    # Risk categorization
    risk = categorize_risk(avg_probability)
    
    return {
        'random_forest': {'prediction': int(rf_pred), 'probability': float(rf_proba)},
        'logistic_regression': {'prediction': int(lr_pred), 'probability': float(lr_proba)},
        'svm': {'prediction': int(svm_pred), 'probability': float(svm_proba)},
        'average_probability': float(avg_probability),
        'risk_assessment': risk,
        'timestamp': datetime.now().isoformat()
    }


# ========================
# ROUTES
# ========================

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for predictions
    
    Expects JSON data with patient features
    Returns prediction results
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        # Validate required fields
        required_fields = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Convert string inputs to numbers
        patient_data = {
            'age': int(data['age']),
            'sex': int(data['sex']),
            'cp': int(data['cp']),
            'trestbps': int(data['trestbps']),
            'chol': int(data['chol']),
            'fbs': int(data['fbs']),
            'restecg': int(data['restecg']),
            'thalach': int(data['thalach']),
            'exang': int(data['exang']),
            'oldpeak': float(data['oldpeak']),
            'slope': int(data['slope']),
            'ca': int(data['ca']),
            'thal': int(data['thal'])
        }
        
        # Make prediction
        result = predict_disease(patient_data)
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/info')
def api_info():
    """Return model and application information"""
    return jsonify({
        'application': 'Heart Disease Prediction System',
        'version': '1.0.0',
        'models': ['Random Forest', 'Logistic Regression', 'SVM'],
        'features': feature_names,
        'accuracy': {
            'random_forest': '87%',
            'logistic_regression': '82%',
            'svm': '85%'
        },
        'disclaimer': 'This is an educational tool only. Not for actual medical diagnosis.'
    }), 200


@app.route('/api/health')
def health_check():
    """Health check endpoint for deployment monitoring"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()}), 200


# ========================
# ERROR HANDLERS
# ========================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Resource not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


# ========================
# MAIN
# ========================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏥 Heart Disease Prediction - Flask Application")
    print("="*60)
    print("\n🌐 Starting Flask server...")
    print("📍 Open http://localhost:5000 in your browser")
    print("\n💡 API Endpoints:")
    print("   POST /api/predict - Make predictions")
    print("   GET /api/info - Get application info")
    print("   GET /api/health - Health check")
    print("\n✋ Press Ctrl+C to stop the server\n")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)