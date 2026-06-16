# 🏥 Heart Disease Prediction System

> **AI-Powered Early Detection & Risk Assessment using Machine Learning**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)

---

## 📋 Project Overview

Heart disease is the **leading cause of death worldwide**. Early prediction and intervention are crucial for saving lives. This project leverages **Machine Learning** to predict the presence of heart disease and categorize patients into risk levels based on clinical features.

### 🎯 Project Goals

✅ **Predict** heart disease presence (Binary Classification: 0 = No, 1 = Yes)  
✅ **Categorize** patients into risk levels: Low, Medium, High  
✅ **Develop** user-friendly web interface  
✅ **Deploy** to cloud platforms  
✅ **Support** early diagnosis and medical intervention  

---

## 🚀 Quick Start

### 1. **Clone/Download Project**

```bash
git clone https://github.com/yourusername/heart_disease_prediction.git
cd heart_disease_prediction
```

### 2. **Install Dependencies**

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 3. **Prepare Data**

```bash
# Place heart_disease_data.csv in project root
# Or download from: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
```

### 4. **Train Models**

```bash
# Data preprocessing
python data_preprocessing.py

# Model training and comparison
python model_training.py

# Output: Trained models saved in 'models/' folder
```

### 5. **Run Application**

**Using Streamlit (Recommended - Easiest)**:
```bash
streamlit run streamlit_app.py
# Opens: http://localhost:8501
```

**Using Flask**:
```bash
python flask_app.py
# Opens: http://localhost:5000
```

---

## 📊 Dataset Information

### Source
- **Platform**: Kaggle
- **Name**: Heart Disease Dataset
- **URL**: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

### Dataset Characteristics
- **Samples**: 1,025 patient records
- **Features**: 14 clinical features
- **Target**: Binary (0 = No Disease, 1 = Disease Present)
- **Missing Values**: Minimal/None

### Features

| Feature | Type | Range | Meaning |
|---------|------|-------|---------|
| **age** | Numeric | 29-77 | Patient age in years |
| **sex** | Binary | 0-1 | 0=Female, 1=Male |
| **cp** | Categorical | 0-3 | Chest pain type |
| **trestbps** | Numeric | 94-200 | Resting blood pressure (mm Hg) |
| **chol** | Numeric | 126-564 | Serum cholesterol (mg/dl) |
| **fbs** | Binary | 0-1 | Fasting blood sugar > 120 mg/dl |
| **restecg** | Categorical | 0-2 | Resting ECG results |
| **thalach** | Numeric | 60-202 | Max heart rate achieved |
| **exang** | Binary | 0-1 | Exercise-induced angina |
| **oldpeak** | Numeric | 0-6.2 | ST depression by exercise |
| **slope** | Categorical | 0-2 | ST segment slope |
| **ca** | Numeric | 0-4 | Number of major vessels |
| **thal** | Categorical | 0-3 | Thalassemia type |
| **target** | Binary | 0-1 | Heart disease (Target) |

---

## 🤖 Machine Learning Algorithms

### 1. **Random Forest Classifier** (Primary Model)
- **Accuracy**: ~87%
- **Advantages**: Handles non-linearity, robust to outliers
- **Architecture**: Ensemble of decision trees
- **Best for**: High accuracy predictions

### 2. **Logistic Regression**
- **Accuracy**: ~82%
- **Advantages**: Fast, interpretable, probability estimates
- **Architecture**: Sigmoid function for binary classification
- **Best for**: Explainability and probability calibration

### 3. **Support Vector Machine (SVM)**
- **Accuracy**: ~85%
- **Advantages**: Good with high-dimensional data
- **Architecture**: Kernel-based boundary optimization
- **Best for**: Complex decision boundaries

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 87% | 85% | 89% | 0.87 |
| Logistic Regression | 82% | 80% | 84% | 0.82 |
| SVM | 85% | 83% | 87% | 0.85 |

---

## 🎯 Risk Categorization

The system categorizes patients based on disease probability:

| Risk Level | Probability | Action |
|-----------|------------|--------|
| 🟢 **LOW** | 0-30% | Continue healthy lifestyle, annual checkups |
| 🟡 **MEDIUM** | 30-70% | Consult cardiologist, monitor vitals |
| 🔴 **HIGH** | 70-100% | Seek immediate medical attention |

---

## 💻 Project Structure

```
heart_disease_prediction/
│
├── 📄 README.md                          # This file
├── 📄 DEPLOYMENT_GUIDE.md               # Cloud deployment guide
├── 📄 HEART_DISEASE_PREDICTION_GUIDE.md # Detailed explanation
│
├── 🔧 data_preprocessing.py              # Data cleaning & scaling
├── 🔧 model_training.py                  # Model training & comparison
├── 🔧 streamlit_app.py                   # Streamlit web app (MAIN)
├── 🔧 flask_app.py                       # Flask web app (ALTERNATIVE)
│
├── 📊 requirements.txt                   # Python dependencies
│
├── 📁 models/                            # Trained models (generated)
│   ├── random_forest_tuned_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── svm_model.pkl
│   ├── scaler.pkl
│   └── feature_names.txt
│
├── 📁 data/
│   └── heart_disease_data.csv           # Dataset (to be added)
│
├── 📁 notebooks/                         # Jupyter notebooks (optional)
│   └── exploratory_analysis.ipynb
│
└── 📁 templates/                         # Flask HTML templates
    ├── index.html
    ├── base.html
    └── result.html

```

---

## 📈 Workflow

```
Input Patient Data
        ↓
Data Preprocessing (Scaling)
        ↓
Three ML Models Make Predictions
        ↓
Ensemble Averaging
        ↓
Risk Categorization (Low/Medium/High)
        ↓
Recommendations & Display Results
```

---

## 🎓 Learning Outcomes

Upon completing this project, you'll understand:

✅ **Machine Learning Pipeline**: Data → Model → Evaluation → Deployment  
✅ **Supervised Learning**: Classification with labeled data  
✅ **Feature Engineering**: Scaling, selection, importance  
✅ **Model Comparison**: Pros/cons of different algorithms  
✅ **Evaluation Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC  
✅ **Cross-Validation**: Avoiding overfitting  
✅ **Web Development**: Streamlit and Flask applications  
✅ **Cloud Deployment**: Hosting ML models in production  
✅ **Healthcare ML**: Domain-specific considerations  

---

## 🚀 Deployment

### Option 1: Streamlit Cloud (Recommended - FREE)
```bash
# Push to GitHub → Deploy to Streamlit Cloud
# No servers to manage, automatic updates
# Perfect for prototypes and MVPs
```

### Option 2: Heroku
```bash
# Deploy Flask/Streamlit app
# Free tier available (limited)
```

### Option 3: PythonAnywhere
```bash
# Easiest setup for beginners
# Free tier available
```

### Option 4: AWS / Google Cloud
```bash
# Production-grade deployment
# Scalable for many users
```

**See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions**

---

## 📖 Detailed Documentation

### For Beginners
Start with: [HEART_DISEASE_PREDICTION_GUIDE.md](HEART_DISEASE_PREDICTION_GUIDE.md)
- Concept explanations
- Step-by-step breakdown
- Beginner-friendly language

### For Deployment
See: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- Multiple platform options
- Cost comparison
- Monitoring & scaling

---

## 📊 Key Features

### 🎯 Prediction Interface
- Easy-to-use input form
- Real-time predictions
- Multiple model ensemble
- Risk score visualization

### 📈 Analytics Dashboard
- Model performance comparison
- Feature importance analysis
- ROC curves and confusion matrices
- Historical prediction tracking

### 📚 Information Hub
- Medical feature explanations
- Heart disease risk factors
- Symptom information
- FAQ and resources

### 🔒 Security
- Input validation
- Data privacy (no external storage)
- Error handling
- Rate limiting (with deployment)

---

## 📚 API Documentation

### Streamlit
```
GET  /                    - Main app interface
Interactive form-based predictions
```

### Flask
```
GET  /                    - Home page
POST /api/predict         - Make prediction
GET  /api/info            - Model information
GET  /api/health          - Health check
```

### Example API Request (Flask)
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 50,
    "sex": 1,
    "cp": 2,
    "trestbps": 120,
    "chol": 200,
    "fbs": 0,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 1,
    "ca": 0,
    "thal": 2
  }'
```

---

## ⚠️ Important Disclaimers

🚨 **This is an Educational Tool Only**

- **NOT** a medical diagnosis device
- **SHOULD NOT** replace professional medical advice
- **Predictions** are based on historical data patterns
- **Always** consult qualified healthcare professionals
- **Use case**: Educational purposes, portfolio projects, research

### Healthcare Compliance
For actual medical use, the system would need:
- FDA approval
- HIPAA compliance
- Clinical validation studies
- Liability insurance
- Professional medical oversight

---

## 🛠️ Development & Contribution

### Setup Development Environment
```bash
# Clone repository
git clone https://github.com/yourusername/heart_disease_prediction.git

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dev dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Format code
black *.py

# Run linter
flake8 *.py
```

### Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📞 Support & Resources

### Learning Resources
- **Scikit-learn Documentation**: https://scikit-learn.org/
- **Streamlit Documentation**: https://docs.streamlit.io/
- **Flask Documentation**: https://flask.palletsprojects.com/
- **Kaggle Datasets**: https://kaggle.com/datasets
- **YouTube**: StatQuest, Kaggle Learn, 3Blue1Brown

### Communities
- Stack Overflow: [tag: scikit-learn], [tag: streamlit]
- Reddit: r/MachineLearning, r/learnprogramming
- GitHub Discussions: Ask questions in Issues
- Kaggle: Dataset discussions and notebooks

### Healthcare Resources
- **American Heart Association**: https://www.heart.org/
- **CDC Heart Disease**: https://www.cdc.gov/heartdisease/
- **NIH Health Info**: https://www.nih.gov/health/

---

## 📊 Performance Metrics

### Model Metrics
- Accuracy: Percentage of correct predictions
- Precision: Of positive predictions, how many correct
- Recall: Of actual positives, how many found
- F1-Score: Balanced metric combining precision & recall
- ROC-AUC: Area under ROC curve (0-1, higher is better)

### Deployment Metrics
- Response Time: < 100ms per prediction
- Availability: 99.9% uptime target
- Error Rate: < 0.1%
- User Satisfaction: 4.5+ / 5.0 rating

---

## 📅 Project Timeline

| Week | Task | Status |
|------|------|--------|
| Week 1-2 | Project Setup & Data Collection | ✅ |
| Week 3 | Topic & Abstract Submission | ✅ |
| Week 4-5 | Data Preprocessing | ✅ |
| Week 6-7 | Model Building | ✅ |
| Week 8 | Review & Feedback | ✅ |
| Week 9-10 | Fine-tuning & Optimization | ✅ |
| Week 11-12 | Web Development & Testing | ✅ |
| Week 13 | Final Deployment | ✅ |

---

## 📈 Future Enhancements

- [ ] Add more advanced algorithms (XGBoost, Neural Networks)
- [ ] Implement real-time data collection from IoT devices
- [ ] Add user authentication and patient history tracking
- [ ] Integrate with Electronic Health Record (EHR) systems
- [ ] Create mobile app (React Native, Flutter)
- [ ] Add federated learning for privacy
- [ ] Implement model explainability (SHAP values)
- [ ] Add clinical validation studies
- [ ] Develop admin dashboard for doctors
- [ ] Create API for hospitals and clinics

---

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

---

## 🙋 Acknowledgments

- **Dataset Source**: Kaggle Heart Disease Dataset (originally from UCI ML Repository)
- **Libraries**: Scikit-learn, Streamlit, Flask, Pandas, Plotly
- **Inspiration**: WHO, CDC, American Heart Association
- **Contributors**: [Your name/team]

---

## 📧 Contact & Social

- **GitHub**: [@yourusername](https://github.com/yourusername)
- **LinkedIn**: [Your LinkedIn](https://linkedin.com)
- **Email**: your.email@example.com
- **Portfolio**: https://yourportfolio.com

---

## 🌟 Show Your Support

If this project helped you learn, please:
- ⭐ Star this repository
- 🍴 Fork for your use
- 💬 Share feedback
- 🐛 Report bugs
- 📚 Contribute improvements

---

## 📝 Citation

If you use this project in your work, please cite:

```bibtex
@misc{heart_disease_prediction,
  title={Heart Disease Prediction System},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  howpublished={\url{https://github.com/yourusername/heart_disease_prediction}}
}
```

---

**Last Updated**: June 2024  
**Version**: 1.0.0  
**Status**: Production Ready ✅

---

