
import streamlit as st
import pandas as pd
import joblib
from custom_knn import CustomKNN   # IMPORTANT: needed for unpickling

st.title("🚢 Titanic Survival Prediction (Custom KNN)")

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("Jawad-Titanic-main/custom_knn_titanic.pkl")

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Enter Passenger Details")

age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=50.0)

pclass = st.selectbox("Passenger Class", [1, 2, 3])
p1 = 1 if pclass == 1 else 0
p2 = 1 if pclass == 2 else 0
p3 = 1 if pclass == 3 else 0

family = st.number_input("Family Size", min_value=0, max_value=10, value=0)

# -----------------------------
# Create input dataframe (MUST match training)
# -----------------------------
input_data = pd.DataFrame([{
    "Age": age,
    "Fare": fare,
    "Pclass_1": p1,
    "Pclass_2": p2,
    "Pclass_3": p3,
    "Family_size": family
}])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success("✅ Passenger Survived")
    else:
        st.error("❌ Passenger Did NOT Survive")
