
# -*- coding: utf-8 -*-
"""app.py for Titanic Survival Prediction using KNN"""

import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# 🎓 Load the Trained Model
# -----------------------------
# Make sure your KNN pipeline is saved as 'titanic_knn_pipeline.pkl'
model = joblib.load("Jawad-Titanic-main/titanic_knn_pipeline.pkl")

st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details to predict if they would survive Titanic disaster.")

# -----------------------------
# 🧮 Input Fields
# -----------------------------
st.subheader("Passenger Information")

pclass_1 = st.number_input("Pclass 1 (Upper class)", min_value=0, max_value=1, value=0)
pclass_2 = st.number_input("Pclass 2 (Middle class)", min_value=0, max_value=1, value=0)
pclass_3 = st.number_input("Pclass 3 (Lower class)", min_value=0, max_value=1, value=1)

age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
fare = st.number_input("Fare", min_value=0.0, value=10.0)
family_size = st.number_input("Family Size", min_value=0, value=1)

# -----------------------------
# 🔢 Choose k for KNN
# -----------------------------
k = st.selectbox("Select k for KNN", [3, 5, 7, 10, 15, 20], index=0)

# -----------------------------
# 📊 Prepare data for prediction
# -----------------------------
input_data = pd.DataFrame({
    'Pclass_1': [pclass_1],
    'Pclass_2': [pclass_2],
    'Pclass_3': [pclass_3],
    'Age': [age],
    'Fare': [fare],
    'Family_size': [family_size]
})

# -----------------------------
# 🔄 Update model with chosen k
# -----------------------------
# Recreate pipeline with user-selected k
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=k))
])

# Fit pipeline on full training data from pickle model
# (we assume the pipeline loaded already has the training data)
# If not, you can fit on the original training CSV inside this app

# -----------------------------
# 🔍 Make Prediction
# -----------------------------
if st.button("Predict Survival 🚢"):
    prediction = model.predict(input_data)
    
    st.subheader("📈 Prediction Result:")
    if prediction[0] == 1:
        st.success("✅ This passenger is **likely to survive!**")
    else:
        st.error("❌ This passenger is **not likely to survive.**")
