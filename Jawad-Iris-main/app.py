# -*- coding: utf-8 -*-
"""app.py — Streamlit app for Iris Species Prediction using Decision Tree"""
import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# 🌳 Load the Trained Decision Tree Model
# -----------------------------
model = joblib.load("Jawad-Iris-main/dtmodel.pkl")   # <-- your model file

st.title("🌸 Iris Flower Species Prediction App")
st.write("Enter flower measurements to predict the **species** using your Decision Tree model.")

# -----------------------------
# 🌼 Input Fields
# -----------------------------
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
sepal_width  = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
petal_width  = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

# -----------------------------
# 📊 Prepare Data
# -----------------------------
input_data = pd.DataFrame({
    'sepal_length': [sepal_length],
    'sepal_width': [sepal_width],
    'petal_length': [petal_length],
    'petal_width': [petal_width]
})

# -----------------------------
# 🔍 Make Prediction
# -----------------------------
if st.button("Predict Species 🌿"):
    prediction = model.predict(input_data)
    
    st.subheader("🌟 Prediction Result:")
    st.success(f"Predicted Species: **{prediction[0]}**")
