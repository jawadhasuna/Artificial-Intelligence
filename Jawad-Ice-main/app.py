# -*- coding: utf-8 -*-
"""app.py — Streamlit app for Ice Cream Revenue Prediction using Decision Tree"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# 🌳 Load the Trained Decision Tree Model
# -----------------------------
model = joblib.load("Jawad-Ice-main/dtmodel.pkl")  # <-- your saved model

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
    
# -----------------------------
# 📈 Optional: Show a chart comparing test data (if available)
# -----------------------------
if st.checkbox("Show sample dataset plot"):
    try:
        df = pd.read_csv("icecream_sales.csv")
        import matplotlib.pyplot as plt
        import numpy as np

        X = df["Temperature"].values.reshape(-1,1)
        y = df["Revenue"].values

        X_grid = np.arange(min(X), max(X), 0.01).reshape(-1,1)
        y_pred_grid = model.predict(X_grid)

        plt.scatter(X, y, color='red', label='Actual Revenue')
        plt.plot(X_grid, y_pred_grid, color='blue', label='Decision Tree Prediction')
        plt.xlabel('Temperature')
        plt.ylabel('Revenue')
        plt.title('Decision Tree Regression')
        plt.legend()
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
