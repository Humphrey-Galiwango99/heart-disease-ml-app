import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load("random_forest_best.joblib")
scaler = joblib.load("scaler.joblib")

st.title("Heart Disease Risk Predictor (Uganda)")

# ---- Collect inputs ----
age = st.number_input("Age (years)", 18, 100, 40)
gender = st.selectbox("Gender", ["Female", "Male"])
ap_hi = st.number_input("Systolic BP (ap_hi)", 80, 250, 120)
ap_lo = st.number_input("Diastolic BP (ap_lo)", 40, 160, 80)
chol = st.selectbox("Cholesterol", ["Normal","Above Normal","Well Above Normal"])
gluc = st.selectbox("Glucose", ["Normal","Above Normal","Well Above Normal"])
smoke = st.selectbox("Smoker", [0,1])
alco = st.selectbox("Alcohol Intake", [0,1])
active = st.selectbox("Physically Active", [0,1])
height = st.number_input("Height (cm)", 120, 220, 170)
weight = st.number_input("Weight (kg)", 30, 200, 70)

# ---- Preprocessing ----
gender_val = 1 if gender == "Male" else 0
chol_val = {"Normal":1, "Above Normal":2, "Well Above Normal":3}[chol]
gluc_val = {"Normal":1, "Above Normal":2, "Well Above Normal":3}[gluc]

# Compute BMI
bmi = weight / ((height/100)**2)

# Age group encoding (manual one-hot)
age_group_26_35 = 1 if 26 <= age <= 35 else 0
age_group_36_45 = 1 if 36 <= age <= 45 else 0
age_group_46_55 = 1 if 46 <= age <= 55 else 0
age_group_56_65 = 1 if 56 <= age <= 65 else 0
age_group_66plus = 1 if age >= 66 else 0

# Build dataframe with correct feature order
cols = ["age_years","gender","ap_hi","ap_lo","cholesterol","gluc","smoke",
        "alco","active","bmi","age_group_26-35","age_group_36-45",
        "age_group_46-55","age_group_56-65","age_group_66+"]

X = pd.DataFrame([[age, gender_val, ap_hi, ap_lo, chol_val, gluc_val,
                   smoke, alco, active, bmi,
                   age_group_26_35, age_group_36_45,
                   age_group_46_55, age_group_56_65, age_group_66plus]], 
                 columns=cols)

# Scale numeric features
X_scaled = scaler.transform(X)

# ---- Prediction ----
pred = model.predict(X_scaled)[0]
proba = model.predict_proba(X_scaled)[0,1]

# ---- Output ----
if pred == 1:
    st.error(f"High risk of heart disease (Probability: {proba:.2f})")
else:
    st.success(f"Low risk of heart disease (Probability: {proba:.2f})")
