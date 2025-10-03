# heart-disease-ml-app
Heart Disease Risk Prediction (Uganda)  This project is my Refactory 2025 Capstone, demonstrating the end-to-end data science pipeline: Data → Cleaning → Modeling → Evaluation → Deployment.  We use patient health indicators (age, blood pressure, cholesterol, BMI, lifestyle habits) to predict the risk of heart disease.
# Problem Statement
Heart disease is one of the fastest-growing health challenges in Uganda due to urbanization, poor diet, and inactivity.
Unfortunately, resources for screening and prevention are limited.
This project builds a machine learning model to help health workers identify high-risk patients early, so that preventive care can be prioritized.

# Dataset
~56,000 patient records

## Features:
Age, Gender, Height, Weight, Blood Pressure (ap_hi/ap_lo), Cholesterol, Glucose
Lifestyle (smoking, alcohol, physical activity)
#### Target: Presence of heart disease (0 = no, 1 = yes)

 # Methodology
Data Cleaning & Preprocessing
Converted age from days → years
Computed BMI from height & weight
Removed unrealistic blood pressure values
Encoded categorical variables

# Modeling
Logistic Regression (baseline, interpretable)
Random Forest (best balance of accuracy & recall)
XGBoost (state-of-the-art boosting method)

# Evaluation Metrics
Accuracy, Precision, Recall, F1-score, ROC-AUC
Focus on Recall (Sensitivity) since missing a sick patient is most dangerous

# Results
Random Forest performed best overall
Key predictors: blood pressure, cholesterol, BMI, age, physical activity
High recall ensures fewer sick patients are missed

# Deployment
The trained model is deployed using Streamlit.
### Live App: Heart Disease Predictor
Users can enter patient details (age, BP, cholesterol, BMI, lifestyle) and receive a risk prediction instantly.

# How to Run Locally
1. Clone repo:
`git clone https://github.com/your-username/heart-disease-ml-app.git
cd heart-disease-ml-app`

2. Install dependencies:
`pip install -r requirements.txt`

3. Run app:
`streamlit run app.py`

### Repository Structure
heart-disease-ml-app/
│── app.py                  # Streamlit app
│── random_forest_best.joblib  # Saved model
│── scaler.joblib           # Preprocessing scaler
│── requirements.txt        # Dependencies
│── README.md               # Documentation

## Future Work
Integrate with mobile health apps
Expand dataset with hospital records in Uganda
Build a clinician dashboard for real-time patient triage

### Author
Humphrey Galiwango
Refactory Uganda – AI/ML Capstone July 2025 Intake
