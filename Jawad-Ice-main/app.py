# -*- coding: utf-8 -*-
"""app.py — Streamlit app for Ice Cream Revenue Prediction using Decision Tree"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# 🌳 Load the Trained Decision Tree Model
# -----------------------------
model = joblib.load("Jawad-Ice-main/tmodel.pkl")  # <-- your saved model

st.title("🍦 Ice Cream Revenue Prediction App")
st.write("Enter the **temperature** to predict ice cream revenue using the Decision Tree model.")

# -----------------------------
# 🌞 Input Field
# -----------------------------
temperature = st.number_input(
    "Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0
)

# -----------------------------
# 📊 Prepare Data
# -----------------------------
input_data = pd.DataFrame({
    'Temperature': [temperature]
})

# -----------------------------
# 🔍 Make Prediction
# -----------------------------
if st.button("Predict Revenue 💰"):
    prediction = model.predict(input_data)
    
    st.subheader("🌟 Prediction Result:")
    st.success(f"Predicted Revenue: **{prediction[0]:.2f}**")
    
