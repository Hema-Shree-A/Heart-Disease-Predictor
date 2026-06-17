"""
HEART DISEASE PREDICTION - STREAMLIT WEB APPLICATION
====================================================
User-friendly interface for heart disease prediction.
Built with Streamlit – interactive UI and easy deployment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
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

st.markdown("""
    <style>
    .main { padding-top: 2rem; }
    .stMetric {
        background-color: #508CA4;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #3498db;
    }
    .success-box {
        padding: 1rem; border-radius: 0.5rem;
        background-color: #d4edda;
        border-left: 5px solid #28a745; color: #155724;
    }
    .danger-box {
        padding: 1rem; border-radius: 0.5rem;
        background-color: #f8d7da;
        border-left: 5px solid #dc3545; color: #721c24;
    }
    .warning-box {
        padding: 1rem; border-radius: 0.5rem;
        background-color: #fff3cd;
        border-left: 5px solid #ffc107; color: #856404;
    }
    </style>
""", unsafe_allow_html=True)


# ========================
# LOAD MODELS & ARTEFACTS
# ========================
@st.cache_resource
def load_models():
    """Load pre-trained models, scaler, and feature list."""
    try:
        rf_model  = joblib.load('models/random_forest_tuned_model.pkl')
        lr_model  = joblib.load('models/logistic_regression_model.pkl')
        svm_model = joblib.load('models/svm_model.pkl')
        scaler    = joblib.load('models/scaler.pkl')

        with open('models/feature_names.txt', 'r') as f:
            feature_names = f.read().strip().split(',')

        return rf_model, lr_model, svm_model, scaler, feature_names
    except FileNotFoundError:
        st.error("⚠️ Models not found! Please run model_training.py first.")
        st.stop()


@st.cache_resource
def load_metrics():
    """Load saved evaluation metrics and feature importances."""
    try:
        with open('models/model_metrics.json', 'r') as f:
            metrics = json.load(f)
        with open('models/feature_importance.json', 'r') as f:
            fi = json.load(f)
        return metrics, fi
    except FileNotFoundError:
        return None, None


rf_model, lr_model, svm_model, scaler, feature_names = load_models()
saved_metrics, saved_fi = load_metrics()


# ========================
# HELPER FUNCTIONS
# ========================
def categorize_risk(probability):
    """Map probability to a risk band."""
    if probability < 0.30:
        return "LOW RISK",    "#28a745", "✅", "Low probability of heart disease"
    elif probability < 0.60:
        return "MEDIUM RISK", "#ffc107", "⚠️", "Moderate risk – consult a doctor"
    else:
        return "HIGH RISK",   "#dc3545", "🚨", "High probability – seek medical attention"


def create_risk_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Risk Score", 'font': {'size': 24}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0,  30], 'color': "#d4edda"},
                {'range': [30, 60], 'color': "#fff3cd"},
                {'range': [60, 100], 'color': "#f8d7da"},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 60,
            }
        },
        number={'suffix': "%", 'font': {'size': 40}}
    ))
    fig.update_layout(height=350)
    return fig


def create_comparison_chart(predictions):
    models = list(predictions.keys())
    probs  = [pred['probability'] * 100 for pred in predictions.values()]
    colors = ["#5fa1e0" if p < 60 else '#dc3545' for p in probs]

    fig = px.bar(
        x=models, y=probs,
        labels={'x': 'Model', 'y': 'Disease Probability (%)'},
        title='Prediction Comparison Across Models',
        text=[f'{p:.1f}%' for p in probs]
    )
    fig.update_traces(marker_color=colors, textposition='outside')
    fig.update_layout(height=400, showlegend=False)
    return fig


def create_feature_importance_chart(feature_importance):
    top = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    features, importances = zip(*top)

    fig = px.bar(
        x=importances, y=features, orientation='h',
        labels={'x': 'Importance Score', 'y': 'Feature'},
        title='Top 10 Most Important Features (Random Forest)',
        color=importances,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=400, showlegend=False)
    return fig


def build_input_vector(patient_data, feature_names):
    """
    Reconstruct the one-hot-encoded feature vector for a single patient,
    matching the exact column order produced during training.
    """
    # Start with zeros for every column that training produced
    encoded = {feat: 0 for feat in feature_names}

    # ── direct numeric / binary columns ──────────────────────────────────────
    for col in ['age', 'sex', 'trestbps', 'chol', 'fbs',
                'thalach', 'exang', 'oldpeak', 'ca']:
        encoded[col] = patient_data[col]

    # ── one-hot dummies (must match astype(str) encoding used in training) ───
    # BUG FIX #2 ── thal now includes 0 as a valid option (7 rows in training
    # data have thal=0). Previously the selectbox only offered [1,2,3], so
    # thal_0 was always zero at inference, causing a hidden feature mismatch.
    for cat_col in ['cp', 'restecg', 'slope', 'thal']:
        col_name = f"{cat_col}_{str(patient_data[cat_col])}"
        if col_name in encoded:
            encoded[col_name] = 1
        # If the value wasn't seen in training (shouldn't happen with valid UI)
        # the column stays 0, which is the safest fallback.

    # Build a single-row DataFrame with columns in training order
    return pd.DataFrame([encoded])[feature_names]


def get_disease_probability(model_obj, input_scaled):
    """
    Return P(heart disease = 1).

    BUG FIX #1 ── The original code used predict_proba()[0][1], which is
    P(class 1). After the target flip in Data_preprocessing.py, class 1 NOW
    correctly means 'has heart disease', so index [1] is correct.
    Before that fix was applied, [1] gave P(healthy), making every high-risk
    patient appear low-risk.
    """
    actual_model = model_obj.model if hasattr(model_obj, 'model') else model_obj
    return actual_model.predict_proba(input_scaled)[0][1]   # P(disease)


# ========================
# APP HEADER
# ========================
st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1>❤️ Heart Disease Prediction System</h1>
        <p style='font-size: 1.2rem; color: #555;'>
            AI-Powered Early Detection & Risk Assessment<br>
            <span style='font-size: 1rem; color: #888;'>by Hema Shree A</span>
        </p>
    </div>
""", unsafe_allow_html=True)

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
        patient_data['age']      = st.slider("Age (years)", 20, 100, 55, 1)
        patient_data['sex']      = st.radio("Sex", [0, 1],
                                            format_func=lambda x: "Female" if x == 0 else "Male")
        patient_data['cp']       = st.selectbox(
            "Chest Pain Type", [0, 1, 2, 3],
            format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina",
                                   2: "Non-anginal Pain", 3: "Asymptomatic"}[x])
        patient_data['trestbps'] = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 130, 1)

    with col2:
        st.subheader("Medical Measurements")
        patient_data['chol']    = st.slider("Cholesterol Level (mg/dl)", 100, 400, 245, 5)
        patient_data['fbs']     = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1],
                                           format_func=lambda x: "No" if x == 0 else "Yes")
        patient_data['restecg'] = st.selectbox(
            "Resting ECG Results", [0, 1, 2],
            format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality",
                                   2: "LV Hypertrophy"}[x])
        patient_data['thalach'] = st.slider("Max Heart Rate Achieved", 60, 202, 150, 1)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Exercise & Symptoms")
        patient_data['exang']   = st.radio("Exercise-Induced Angina", [0, 1],
                                           format_func=lambda x: "No" if x == 0 else "Yes")
        patient_data['oldpeak'] = st.slider("ST Depression (Exercise)", 0.0, 6.2, 1.0, 0.1)

    with col4:
        st.subheader("Additional Tests")
        patient_data['slope'] = st.selectbox(
            "Slope of ST Segment", [0, 1, 2],
            format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x])
        patient_data['ca']   = st.selectbox("Major Vessels (Fluoroscopy)", [0, 1, 2, 3, 4])

        # ── BUG FIX #2 ── added 0 = "Unknown/Normal" to match training data ──
        patient_data['thal'] = st.selectbox(
            "Thalassemia", [0, 1, 2, 3],
            format_func=lambda x: {0: "Unknown/Normal", 1: "Normal",
                                   2: "Fixed Defect", 3: "Reversible Defect"}[x])

    # ── Prediction button ─────────────────────────────────────────────────────
    if st.button("🔮 Predict Heart Disease Risk", use_container_width=True):

        input_df     = build_input_vector(patient_data, feature_names)

        # Scale continuous columns
        continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
        input_scaled    = input_df.copy()
        input_scaled[continuous_cols] = scaler.transform(input_df[continuous_cols])

        # Collect predictions from all three models
        predictions = {}
        for name, model_obj in [
            ('Random Forest',          rf_model),
            ('Logistic Regression',    lr_model),
            ('Support Vector Machine', svm_model),
        ]:
            actual_model = model_obj.model if hasattr(model_obj, 'model') else model_obj
            prob = get_disease_probability(model_obj, input_scaled)
            predictions[name] = {
                'prediction': actual_model.predict(input_scaled)[0],
                'probability': prob,
            }

        avg_prob = np.mean([p['probability'] for p in predictions.values()])
        risk_level, color, emoji, description = categorize_risk(avg_prob)

        st.success("✅ Prediction Complete!")
        st.markdown(f"""
            <div style='text-align:center; padding:2rem; background-color:#1e293b;
                        border-radius:1rem; color:white; margin:1rem 0;'>
                <h2 style='color:{color}'>{emoji} {risk_level}</h2>
                <h3>{avg_prob*100:.1f}% Probability of Heart Disease</h3>
                <p>{description}</p>
            </div>
        """, unsafe_allow_html=True)

        st.subheader("📊 Model-wise Predictions")
        cols = st.columns(3)
        for idx, (name, pred) in enumerate(predictions.items()):
            r, _, e, _ = categorize_risk(pred['probability'])
            with cols[idx]:
                st.metric(label=name,
                          value=f"{pred['probability']*100:.1f}%",
                          delta=r)

        st.plotly_chart(create_risk_gauge(avg_prob),
                        use_container_width=True,
                        config={'displayModeBar': False})
        st.plotly_chart(create_comparison_chart(predictions),
                        use_container_width=True,
                        config={'displayModeBar': False})

        # Recommendations
        st.subheader("💊 Recommendations")
        if avg_prob < 0.30:
            st.markdown("""
                <div class='success-box'><strong>✅ Low Risk:</strong>
                <ul>
                    <li>Maintain healthy lifestyle and regular exercise (150 min/week)</li>
                    <li>Balanced diet with reduced salt and saturated fats</li>
                    <li>Annual check-ups recommended</li>
                </ul></div>""", unsafe_allow_html=True)
        elif avg_prob < 0.60:
            st.markdown("""
                <div class='warning-box'><strong>⚠️ Medium Risk:</strong>
                <ul>
                    <li>Consult a cardiologist for detailed evaluation</li>
                    <li>Monitor blood pressure and cholesterol regularly</li>
                    <li>Reduce stress through meditation or yoga</li>
                    <li>Limit alcohol, quit smoking</li>
                    <li>Follow a cardiac-friendly diet</li>
                </ul></div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class='danger-box'><strong>🚨 High Risk:</strong>
                <ul>
                    <li><strong>Seek medical attention immediately</strong></li>
                    <li>Schedule urgent consultation with a cardiologist</li>
                    <li>Consider ECG, Troponin, Echocardiogram tests</li>
                    <li>Hospital admission may be necessary</li>
                </ul></div>""", unsafe_allow_html=True)

        st.session_state.last_prediction = {
            'timestamp':  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'age':        patient_data['age'],
            'risk_level': risk_level,
            'probability': avg_prob,
        }


# ========================
# TAB 2: ANALYTICS
# ========================
with tab2:
    st.header("📊 Model Analytics & Performance")

    # BUG FIX #3 ── Use REAL metrics from training, not hardcoded values
    if saved_metrics:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Model Performance Metrics (from actual evaluation)")
            rows = []
            for model_name, m in saved_metrics.items():
                rows.append({
                    'Model':     model_name,
                    'Accuracy':  round(m['accuracy'],  4),
                    'Precision': round(m['precision'], 4),
                    'Recall':    round(m['recall'],    4),
                    'F1-Score':  round(m['f1_score'],  4),
                })
            df_metrics = pd.DataFrame(rows)
            st.dataframe(df_metrics, use_container_width=True)

        with col2:
            st.subheader("Model Comparison")
            fig = px.bar(
                df_metrics, x='Model',
                y=['Accuracy', 'Precision', 'Recall'],
                barmode='group',
                title='Performance Metrics Comparison',
                labels={'value': 'Score', 'variable': 'Metric'}
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Run model_training.py first to generate model_metrics.json")

    st.subheader("🎯 Feature Importance (Tuned Random Forest)")
    if saved_fi:
        st.plotly_chart(
            create_feature_importance_chart(saved_fi),
            use_container_width=True,
            config={'displayModeBar': False}
        )
    else:
        st.info("Feature importance will appear after training.")


# ========================
# TAB 3: INFORMATION
# ========================
with tab3:
    st.header("📚 Medical Information & Features")
    col1, col2 = st.columns(2)

    with col1:
        st.write("""
        **Modifiable Risk Factors:**
        - 🚭 Smoking
        - 🍔 Poor Diet
        - 📺 Sedentary Lifestyle
        - 🩸 High Blood Pressure
        - 🍬 Diabetes
        """)
        st.write("""
        **Key Features Explained:**
        - **thalach** – Maximum heart rate during stress test. *Lower* values in older
          patients often indicate reduced cardiac capacity.
        - **oldpeak** – ST depression after exercise. *Higher* values indicate more
          cardiac stress.
        - **ca** – Number of major vessels visible on fluoroscopy. More blocked vessels
          = higher risk.
        - **thal** – Thalassemia type. Reversible defect is associated with higher risk.
        """)

    with col2:
        st.write("""
        **Warning Signs:**
        - 💔 Chest pain / discomfort
        - 😮‍💨 Shortness of breath
        - 😰 Cold sweats, nausea
        - 💫 Lightheadedness
        """)
        st.write("""
        **High-Risk Profile (use these values to test HIGH RISK):**
        - Age: 65+, Male
        - Chest Pain: Typical Angina (0)
        - Max Heart Rate: 100–130 (low)
        - Exercise Angina: Yes
        - ST Depression: 3.0+
        - Major Vessels: 2–4
        - Thalassemia: Reversible Defect
        """)


# ========================
# TAB 4: FAQ
# ========================
with tab4:
    st.header("❓ Frequently Asked Questions")
    
    faq_items = [
        ("What is the accuracy of this model?", 
         "The Random Forest model achieves 87% accuracy on the test set. However, this is a supportive tool and should not replace medical diagnosis."),
        
        ("How is the prediction probability calculated?",
         "The system uses three machine learning models and takes the average probability. Each model independently analyzes the patient data using different algorithms."),
        
        ("What does risk level mean?",
         "LOW (0-30%): Low probability of disease. MEDIUM (30-70%): Moderate risk, consult doctor. HIGH (70-100%): High probability, seek immediate attention."),
        
        ("Can this replace a doctor?",
         "NO! This is a supportive diagnostic tool only. Always consult qualified healthcare professionals for accurate diagnosis and treatment."),
        
        ("Why are there different predictions?",
         "Different algorithms make different decisions. The average provides more robust prediction. Multiple opinions help reduce bias."),
        
        ("How much data does the model use?",
         "The model was trained on 1025 patient records with 14 clinical features from the Kaggle Heart Disease Dataset."),
        
        ("What features are most important?",
         "Chest pain type, maximum heart rate, ST depression, resting blood pressure, and age are the most important factors."),
        
        ("Is my data secure?",
         "This application processes data locally. No data is sent to external servers. However, always consult official medical institutions for sensitive health data."),
    ]
    
    for question, answer in faq_items:
        with st.expander(question):
            st.write(answer)
 


# ========================
# FOOTER
# ========================
st.markdown("""
    <hr style='margin-top:3rem;'>
    <div style='text-align:center; color:#888; font-size:0.9rem; padding:1rem;'>
        <p>⚠️ <strong>Disclaimer</strong>: Educational AI tool for demonstration only.
        Always consult a qualified medical professional.</p>
    </div>
""", unsafe_allow_html=True)