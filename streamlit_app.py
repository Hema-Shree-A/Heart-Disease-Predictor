"""
HEART DISEASE PREDICTION - STREAMLIT WEB APPLICATION
====================================================
User-friendly interface for heart disease prediction
Built with Streamlit - Easy deployment and interactive UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ========================
# PAGE CONFIGURATION
# ========================
st.set_page_config(
    page_title="❤️ Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stMetric {
        background-color:#508CA4;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #3498db;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        color: #155724;
    }
    .danger-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        color: #721c24;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        color: #856404;
    }
    </style>
""", unsafe_allow_html=True)

# ========================
# LOAD MODELS & SCALER
# ========================
@st.cache_resource
def load_models():
    """Load pre-trained models and scaler"""
    try:
        rf_model = joblib.load('models/random_forest_tuned_model.pkl')
        lr_model = joblib.load('models/logistic_regression_model.pkl')
        svm_model = joblib.load('models/svm_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        
        with open('models/feature_names.txt', 'r') as f:
            feature_names = f.read().strip().split(',')
        
        return rf_model, lr_model, svm_model, scaler, feature_names
    except FileNotFoundError:
        st.error("⚠️ Models not found! Please train models first using model_training.py")
        st.stop()

# Load models
rf_model, lr_model, svm_model, scaler, feature_names = load_models()

# ========================
# HELPER FUNCTIONS
# ========================
def categorize_risk(probability):
    """
    Categorize risk level based on prediction probability
    
    Args:
        probability: Probability of heart disease (0-1)
    
    Returns:
        tuple: (risk_level, color, emoji, description)
    """
    if probability < 0.3:
        return "LOW RISK", "#28a745", "✅", "Low probability of heart disease"
    elif probability < 0.7:
        return "MEDIUM RISK", "#ffc107", "⚠️", "Moderate risk - Consult doctor"
    else:
        return "HIGH RISK", "#dc3545", "🚨", "High probability - Seek medical attention"


def create_risk_gauge(probability):
    """Create a gauge chart for risk visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        title={'text': "Risk Score", 'font': {'size': 24}}, 
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "#d4edda"},
                {'range': [30, 70], 'color': "#fff3cd"},
                {'range': [70, 100], 'color': "#f8d7da"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        },
        number={
            'suffix': "%", 
            'font': {'size': 40}
        }
    ))
    fig.update_layout(height=350)
    return fig


def create_comparison_chart(predictions):
    """Create a bar chart comparing predictions from all models"""
    models = list(predictions.keys())
    probs = [pred['probability'] * 100 for pred in predictions.values()]
    
    fig = px.bar(
        x=models,
        y=probs,
        labels={'x': 'Models', 'y': 'Disease Probability'},
        title='Prediction Comparison Across Models',
        color=['#28a745' if p < 50 else '#dc3545' for p in probs],
        text=[f'{p:.1f}%' for p in probs]
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(height=400)
    return fig

def create_feature_importance_chart(feature_importance):
    """Create a horizontal bar chart for feature importance"""
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    features, importances = zip(*top_features)
    
    fig = px.bar(
        x=importances,
        y=features,
        labels={'x': 'Importance Score', 'y': 'Features'},
        title='Top 10 Most Important Features (Random Forest)',
        color=importances,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=400, showlegend=False)
    return fig


# ========================
# MAIN APP LAYOUT
# ========================
# Header
st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1>❤️ Heart Disease Prediction System</h1>
        <p style='font-size: 1.2rem; color: #555;'>
            AI-Powered Early Detection & Risk Assessment
            <h2 style='font-size: 1rem; color: #888;'>by Hema Shree A</h2>
        </p>
    </div>
""", unsafe_allow_html=True)
# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📊 Analytics", "📚 Information", "❓ FAQ"])

# ========================
# TAB 1: PREDICTION
# ========================
with tab1:
    st.header("Patient Information & Prediction")
    
    col1, col2 = st.columns(2)
    
    patient_data = {}
    
    with col1:
        st.subheader("Basic Information")
        patient_data['age'] = st.slider("Age (years)", min_value=20, max_value=100, value=50, step=1)
        patient_data['sex'] = st.radio("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        patient_data['cp'] = st.selectbox("Chest Pain Type", 
                                          options=[0, 1, 2, 3],
                                          format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina", 
                                                                 2: "Non-anginal Pain", 3: "Asymptomatic"}[x])
        patient_data['trestbps'] = st.slider("Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=120, step=1)
    
    with col2:
        st.subheader("Medical Measurements")
        patient_data['chol'] = st.slider("Cholesterol Level (mg/dl)", min_value=100, max_value=400, value=200, step=5)
        patient_data['fbs'] = st.radio("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], 
                                       format_func=lambda x: "No" if x == 0 else "Yes")
        patient_data['restecg'] = st.selectbox("Resting ECG Results",
                                              options=[0, 1, 2],
                                              format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}[x])
        patient_data['thalach'] = st.slider("Max Heart Rate Achieved", min_value=60, max_value=200, value=150, step=1)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Exercise & Symptoms")
        patient_data['exang'] = st.radio("Exercise-Induced Angina", options=[0, 1],
                                        format_func=lambda x: "No" if x == 0 else "Yes")
        patient_data['oldpeak'] = st.slider("ST Depression Induced by Exercise", min_value=0.0, max_value=6.2, value=1.0, step=0.1)
    
    with col4:
        st.subheader("Additional Tests")
        patient_data['slope'] = st.selectbox("Slope of ST Segment",
                                            options=[0, 1, 2],
                                            format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x])
        patient_data['ca'] = st.selectbox("Major Vessels (Fluoroscopy)", options=[0, 1, 2, 3, 4])
        patient_data['thal'] = st.selectbox("Thalassemia",
                                           options=[0, 1, 2, 3],
                                           format_func=lambda x: {0: "Normal", 1: "Fixed Defect", 
                                                                 2: "Reversible Defect", 3: "Unknown"}[x])
    
    # Make prediction button
    if st.button("🔮 Predict Heart Disease Risk", key="predict_btn", width="stretch"):        
        input_df = pd.DataFrame([patient_data])
        input_df = input_df[feature_names]
        input_scaled = scaler.transform(input_df)
        
        predictions = {
            'Random Forest': {
                'prediction': rf_model.predict(input_scaled)[0],
                'probability': rf_model.predict_proba(input_scaled)[0][1]
            },
            'Logistic Regression': {
                'prediction': lr_model.predict(input_scaled)[0],
                'probability': lr_model.predict_proba(input_scaled)[0][1]
            },
            'Support Vector Machine': {
                'prediction': svm_model.predict(input_scaled)[0],
                'probability': svm_model.predict_proba(input_scaled)[0][1]
            }
        }
        
        avg_probability = np.mean([p['probability'] for p in predictions.values()])
        st.success("✅ Prediction Complete!")
        risk_level, color, emoji, description = categorize_risk(avg_probability)
        
        st.markdown(f"""
            <div style='text-align: center; padding: 2rem; background-color: #1e293b; 
                        border-radius: 1rem; color: white; margin: 1rem 0;'>
                <h2>{emoji} {risk_level}</h2>
                <h3>{avg_probability*100:.1f}% Probability of Heart Disease</h3>
                <p>{description}</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.subheader("📊 Model-wise Predictions")
        col_pred1, col_pred2, col_pred3 = st.columns(3)
        
        for idx, (model_name, pred) in enumerate(predictions.items()):
            risk_cat, _, emoji_risk, desc = categorize_risk(pred['probability'])
            col = col_pred1 if idx == 0 else col_pred2 if idx == 1 else col_pred3
            with col:
                st.metric(label=model_name, value=f"{pred['probability']*100:.1f}%", delta=risk_cat)
        
        st.plotly_chart(create_risk_gauge(avg_probability), width="stretch", config={'displayModeBar':False })
        st.plotly_chart(create_comparison_chart(predictions), width="stretch", config={'displayModeBar':False })
        
        st.subheader("💊 Recommendations")
        if avg_probability < 0.3:
            st.markdown("""
                <div class='success-box'><strong>✅ Low Risk:</strong>
                    <ul>
                        <li>Continue maintaining a healthy lifestyle</li>
                        <li>Regular exercise (150 min/week moderate intensity)</li>
                        <li>Balanced diet with reduced salt and saturated fats</li>
                        <li>Annual check-ups recommended</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        elif avg_probability < 0.7:
            st.markdown("""
                <div class='warning-box'><strong>⚠️ Medium Risk:</strong>
                    <ul>
                        <li>Consult a cardiologist for detailed evaluation</li>
                        <li>Monitor blood pressure and cholesterol regularly</li>
                        <li>Reduce stress through meditation or yoga</li>
                        <li>Limit alcohol consumption</li>
                        <li>Quit smoking if applicable</li>
                        <li>Follow a cardiac-friendly diet</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class='danger-box'><strong>🚨 High Risk:</strong>
                    <ul>
                        <li><strong>Seek medical attention immediately</strong></li>
                        <li>Schedule urgent consultation with a cardiologist</li>
                        <li>Consider additional diagnostic tests (ECG, Troponin, Echocardiogram)</li>
                        <li>May require medication or intervention</li>
                        <li>Hospital admission may be necessary</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        st.session_state.last_prediction = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'age': patient_data['age'],
            'risk_level': risk_level,
            'probability': avg_probability
        }


# ========================
# TAB 2: ANALYTICS
# ========================
with tab2:
    st.header("📊 Model Analytics & Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance Metrics")
        metrics_data = {
            'Model': ['Random Forest', 'Logistic Regression', 'SVM'],
            'Accuracy': [0.87, 0.82, 0.85],
            'Precision': [0.85, 0.80, 0.83],
            'Recall': [0.89, 0.84, 0.87],
            'F1-Score': [0.87, 0.82, 0.85]
        }
        df_metrics = pd.DataFrame(metrics_data)
        # Fixed: Changed use_container_width=True to width="stretch"
        st.dataframe(df_metrics, width="stretch")
    
    with col2:
        st.subheader("Model Comparison")
        fig = px.bar(
            df_metrics,
            x='Model',
            y=['Accuracy', 'Precision', 'Recall'],
            barmode='group',
            title='Performance Metrics Comparison',
            labels={'value': 'Score', 'variable': 'Metric'}
        )
        # Fixed: Changed use_container_width=True to width="stretch"
        st.plotly_chart(fig, width="stretch")
    
    st.subheader("🎯 Feature Importance (Random Forest)")
    feature_importance = {
        'cp': 0.15, 'thalach': 0.14, 'oldpeak': 0.13, 'trestbps': 0.12, 'age': 0.11,
        'chol': 0.10, 'ca': 0.09, 'exang': 0.08, 'slope': 0.07, 'thal': 0.01
    }
    st.plotly_chart(create_feature_importance_chart(feature_importance), width="stretch", config={'displayModeBar':False })
    
    st.subheader("🔍 Model Interpretability")
    st.write("""
    **Random Forest**: Uses ensemble of decision trees for robust predictions
    - Handles non-linear relationships
    - Features like chest pain type and maximum heart rate are most important
    
    **Logistic Regression**: Provides probability-based predictions
    - Highly interpretable
    - Shows direct relationship between features and disease risk
    
    **SVM**: Finds optimal boundary between disease classes
    - Works well for binary classification
    - Memory efficient
    """)


# ========================
# TAB 3: INFORMATION
# ========================
with tab3:
    st.header("📚 Medical Information & Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Heart Disease Risk Factors")
        st.write("""
        **Modifiable Risk Factors:**
        - 🚭 Smoking
        - 🍔 Poor Diet
        - 📺 Sedentary Lifestyle
        - ⚖️ Obesity
        - 😰 Stress
        
        **Non-Modifiable Risk Factors:**
        - 👨‍👩‍👧 Family History
        - 📊 Age
        - 👨 Gender
        """)
    
    with col2:
        st.subheader("Key Symptoms")
        st.write("""
        **Warning Signs:**
        - 💔 Chest pain/discomfort
        - 😮‍💨 Shortness of breath
        - 😴 Fatigue
        - 😖 Dizziness
        - 💦 Cold sweats
        - 😐 Jaw/Neck/Back pain
        
        **Seek Emergency Help If:**
        - Severe chest pain
        - Unable to breathe
        - Loss of consciousness
        """)
    
    st.subheader("Feature Explanations")
    features_info = {
        'Age': 'Patient age in years (higher risk with age)',
        'Sex': 'Gender (1=Male, 0=Female; Males typically at higher risk)',
        'Chest Pain Type': 'Classification of chest pain (0-3 scale)',
        'Resting BP': 'Blood pressure at rest (normal: <120 mm Hg)',
        'Cholesterol': 'Serum cholesterol level (optimal: <200 mg/dl)',
        'Festing Blood Sugar': 'Blood sugar level after fasting >120 mg/dl',
        'Resting ECG': 'Electrocardiogram results at rest',
        'Max Heart Rate': 'Maximum heart rate achieved during exercise',
        'Exercise Angina': 'Angina (chest pain) induced by exercise',
        'Oldpeak': 'ST depression induced by exercise',
        'ST Slope': 'Slope of ST segment in ECG',
        'Major Vessels': 'Number of vessels with >50% stenosis',
        'Thalassemia': 'Blood disorder classification'
    }
    
    with st.expander("📋 Detailed Feature Descriptions"):
        for feature, description in features_info.items():
            st.write(f"**{feature}**: {description}")


# ========================
# TAB 4: FAQ
# ========================
with tab4:
    st.header("❓ Frequently Asked Questions")
    faq_items = [
        ("What is the accuracy of this model?", "The Random Forest model achieves 87% accuracy on the test set. However, this is a supportive tool and should not replace medical diagnosis."),
        ("How is the prediction probability calculated?", "The system uses three machine learning models and takes the average probability. Each model independently analyzes the patient data using different algorithms."),
        ("What does risk level mean?", "LOW (0-30%): Low probability of disease. MEDIUM (30-70%): Moderate risk, consult doctor. HIGH (70-100%): High probability, seek immediate attention."),
        ("Can this replace a doctor?", "NO! This is a supportive diagnostic tool only. Always consult qualified healthcare professionals for accurate diagnosis and treatment."),
        ("Why are there different predictions?", "Different algorithms make different decisions. The average provides more robust prediction. Multiple opinions help reduce bias."),
        ("How much data does the model use?", "The model was trained on 1025 patient records with 14 clinical features from the Kaggle Heart Disease Dataset."),
        ("What features are most important?", "Chest pain type, maximum heart rate, ST depression, resting blood pressure, and age are the most important factors."),
        ("Is my data secure?", "This application processes data locally. No data is sent to external servers. However, always consult official medical institutions for sensitive health data."),
    ]
    for question, answer in faq_items:
        with st.expander(question):
            st.write(answer)


# ========================
# FOOTER
# ========================
st.markdown("""
    <hr style='margin-top: 3rem;'>
    <div style='text-align: center; color: #888; font-size: 0.9rem; padding: 1rem;'>
        <p>❤️ Heart Disease Prediction System</p>
        <p>⚠️ <strong>Disclaimer</strong>: This is an educational AI tool for demonstration purposes only. 
        It should NOT be used for actual medical diagnosis. Always consult qualified medical professionals.</p>
        <p>Built with Streamlit | ML Models: Random Forest, Logistic Regression, SVM</p>
    </div>
""", unsafe_allow_html=True)