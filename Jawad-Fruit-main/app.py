# -*- coding: utf-8 -*-
"""app.py — Streamlit app for Apples vs Oranges Prediction using Random Forest"""

import streamlit as st
import pandas as pd
import joblib
import sklearn

# -----------------------------
# 🍎🍊 App Info
# -----------------------------
st.title("🍎 Apples vs Oranges Classification App")
st.write(
    "Enter the **weight** and **size** of the fruit to predict whether it is an **Apple** or an **Orange** "
    "using a Random Forest model."
)

# Optional: show versions (useful for debugging)
st.caption(f"Streamlit version: {st.__version__}")
st.caption(f"scikit-learn version: {sklearn.__version__}")

# -----------------------------
# 🌳 Load Trained Model
# -----------------------------
model = joblib.load("Jawad-Fruit-main/apples_oranges.pkl")

# -----------------------------
# 🧮 Input Fields
# -----------------------------
weight = st.number_input(
    "Weight (grams)",
    min_value=0.0,
    max_value=500.0,
    value=70.0
)

size = st.number_input(
    "Size",
    min_value=0.0,
    max_value=10.0,
    value=5.0
)

# -----------------------------
# 📊 Prepare Input Data
# -----------------------------
input_data = pd.DataFrame({
    "Weight": [weight],
    "Size": [size]
})

# -----------------------------
# 🔍 Make Prediction
# -----------------------------
if st.button("Predict Fruit 🍏🍊"):
    prediction = model.predict(input_data)[0]

    fruit = "Apple 🍎" if prediction == 1 else "Orange 🍊"

    st.subheader("✅ Prediction Result")
    st.success(f"The predicted fruit is: **{fruit}**")
