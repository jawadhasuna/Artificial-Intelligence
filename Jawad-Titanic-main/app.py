import streamlit as st
import pandas as pd
import joblib

st.title("🚢 Titanic Survival Prediction App")
st.write("This app predicts whether a passenger would survive the Titanic disaster.")

# Load model + scaler
model = joblib.load("Jawad-Titanic-main/titan.pkl")
scaler = joblib.load("Jawad-Titanic-main/scale.pkl")

# User inputs
age = st.number_input("Age", 0.0, 100.0, 30.0)
fare = st.number_input("Fare", 0.0, 600.0, 32.0)
pclass1 = st.selectbox("Pclass 1 (1=yes, 0=no)", [0, 1])
pclass2 = st.selectbox("Pclass 2 (1=yes, 0=no)", [0, 1])
pclass3 = st.selectbox("Pclass 3 (1=yes, 0=no)", [0, 1])
family = st.number_input("Family Size", 0, 10, 1)

# DataFrame
user_data = pd.DataFrame({
    "Age": [age],
    "Fare": [fare],
    "Pclass_1": [pclass1],
    "Pclass_2": [pclass2],
    "Pclass_3": [pclass3],
    "Family_size": [family]
})

if st.button("Predict Survival"):
    # Scale input
    scaled = scaler.transform(user_data)

    # Predict
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    st.subheader("Prediction Result:")
    
    if pred == 1:
        st.success(f"🟩 Passenger would SURVIVE! (Probability: {prob*100:.2f}%)")
    else:
        st.error(f"🟥 Passenger would NOT survive. (Probability: {prob*100:.2f}%)")
