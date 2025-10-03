import streamlit as st
import joblib
import numpy as np

# Load saved model and scaler
model = joblib.load("random_forest_best.joblib")
scaler = joblib.load("scaler.joblib")

st.title("Heart Disease Risk Predictor (Uganda)")
st.write("Enter patient details to check heart disease risk:")

# Input fields
age = st.number_input("Age (years)", 18, 100, 40)
gender = st.selectbox("Gender", ["Female", "Male"])
ap_hi = st.number_input("Systolic BP (ap_hi)", 80, 250, 120)
ap_lo = st.number_input("Diastolic BP (ap_lo)", 40, 160, 80)
chol = st.selectbox("Cholesterol", ["Normal","Above Normal","Well Above Normal"])
gluc = st.selectbox("Glucose", ["Normal","Above Normal","Well Above Normal"])
bmi = st.number_input("BMI", 10, 60, 25)
smoke = st.selectbox("Smoker", [0,1])
alco = st.selectbox("Alcohol Intake", [0,1])
active = st.selectbox("Physically Active", [0,1])

# Preprocess inputs
gender_val = 1 if gender == "Male" else 0
chol_val = {"Normal":1, "Above Normal":2, "Well Above Normal":3}[chol]
gluc_val = {"Normal":1, "Above Normal":2, "Well Above Normal":3}[gluc]

X = np.array([[age, gender_val, ap_hi, ap_lo, chol_val, gluc_val,
               smoke, alco, active, bmi]])

X_scaled = scaler.transform(X)

# Prediction
pred = model.predict(X_scaled)[0]
proba = model.predict_proba(X_scaled)[0,1]

# Output
if pred == 1:
    st.error(f"High risk of heart disease ⚠️ (Probability: {proba:.2f})")
else:
    st.success(f"Low risk of heart disease ✅ (Probability: {proba:.2f})")
