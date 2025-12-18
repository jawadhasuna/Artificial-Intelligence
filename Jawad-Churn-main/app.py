import streamlit as st
import numpy as np
import joblib
from keras.models import load_model

# Load model and scaler
model = load_model("Jawad-Churn-main/churn_ann_model.h5")
scaler = joblib.load("Jawad-Churn-main/scaler.pkl")

st.title("Customer Churn Prediction (ANN)")
st.write("Predict whether a customer will leave the bank")

# User inputs
credit_score = st.number_input("Credit Score", 300, 900, 650)
age = st.number_input("Age", 18, 100, 40)
tenure = st.number_input("Tenure (years)", 0, 10, 3)
balance = st.number_input("Balance", 0.0, 300000.0, 60000.0)
num_products = st.number_input("Number of Products", 1, 4, 2)
has_card = st.selectbox("Has Credit Card?", [0, 1])
is_active = st.selectbox("Is Active Member?", [0, 1])
salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])

# Encoding
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0
gender_male = 1 if gender == "Male" else 0

# Input array
input_data = np.array([[credit_score, age, tenure, balance,
                         num_products, has_card, is_active, salary,
                         geo_germany, geo_spain, gender_male]])

# Scale
input_data = scaler.transform(input_data)

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_data)
    if prediction > 0.5:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is NOT likely to churn")
