import streamlit as st
import pandas as pd
import joblib

# Load files
scaler = joblib.load("Jawad-Titanic-main/ss.pkl")
model = joblib.load("Jawad-Titanic-main/knn.pkl")

st.title("🚢 Titanic Survival Prediction (KNN Model)")

st.write("Fill in the passenger details below:")

# Inputs (same features used during training)
age = st.number_input("Age", 0, 100, 30)
fare = st.number_input("Fare", 0.0, 600.0, 32.2)
p1 = st.selectbox("Pclass 1", [0,1])
p2 = st.selectbox("Pclass 2", [0,1])
p3 = st.selectbox("Pclass 3", [0,1])
family = st.number_input("Family Size", 0, 15, 1)

# Create dataframe
df = pd.DataFrame({
    "Age":[age],
    "Fare":[fare],
    "Pclass_1":[p1],
    "Pclass_2":[p2],
    "Pclass_3":[p3],
    "Family_size":[family]
})

if st.button("Predict"):
    scaled = scaler.transform(df)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    st.subheader("Result:")
    if pred == 1:
        st.success(f"✔ Passenger is LIKELY to survive ({prob*100:.2f}% probability)")
    else:
        st.error(f"✘ Passenger is UNLIKELY to survive ({prob*100:.2f}% probability)")
