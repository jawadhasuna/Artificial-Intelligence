
# -*- coding: utf-8 -*-
"""Jay's Hospital Breast Cancer Prediction App"""

import streamlit as st
import numpy as np
import joblib

# -----------------------------
# 🎯 Load the Trained Model & Scaler
# -----------------------------
model = joblib.load("Jawad-Wisconsin-main/svm_rbf_model.pkl")
scaler = joblib.load("Jawad-Wisconsin-main/scaler.pkl")
selected_features = joblib.load("Jawad-Wisconsin-main/selected_features.pkl")

st.title("🩺 Breast Cancer Prediction App")
st.write("Enter values for the top features to predict whether the cancer is **Malignant** or **Benign**.")

# -----------------------------
# 🧩 Feature Ranges (Guide User)
# -----------------------------
feature_ranges = {
    'radius_mean': (6.0, 30.0),
    'perimeter_mean': (40.0, 190.0),
    'area_mean': (100.0, 2500.0),
    'concavity_mean': (0.0, 0.35),
    'concave points_mean': (0.0, 0.2),
    'radius_worst': (7.0, 36.0),
    'perimeter_worst': (50.0, 250.0),
    'area_worst': (150.0, 4000.0),
    'concavity_worst': (0.0, 0.4),
    'concave points_worst': (0.0, 0.3)
}

# -----------------------------
# 🖊️ Input Fields for Top Features
# -----------------------------
inputs = []
for feature in selected_features:
    min_val, max_val = feature_ranges.get(feature, (0.0, 100.0))
    value = st.number_input(
        f"{feature} (range: {min_val}-{max_val})",
        min_value=float(min_val),
        max_value=float(max_val),
        step=0.01
    )
    inputs.append(value)

user_input = np.array(inputs).reshape(1, -1)
user_input_scaled = scaler.transform(user_input)

# -----------------------------
# 🔍 Make Prediction
# -----------------------------
if st.button("Predict Cancer Type 🩺"):
    prediction = model.predict(user_input_scaled)[0]
    confidence = model.decision_function(user_input_scaled)[0]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error("❌ Malignant (Cancerous)")
    else:
        st.success("✅ Benign (Non-cancerous)")

    st.write(f"Confidence Score: {confidence:.2f}")


