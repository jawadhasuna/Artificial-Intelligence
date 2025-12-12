import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# Load Custom KNN Model
# ---------------------------------------------------------
model = joblib.load("Jawad-Titanic-main/custom_knn_titanic.pkl")   # <-- use your pickle filename


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.title("🚢 Titanic Survival Prediction App (Custom KNN)")
st.write("Enter passenger details to predict whether they **survived or not**.")

# ---------------------------------------------------------
# User Input Fields
# ---------------------------------------------------------
pclass = st.number_input("Passenger Class (1 = Upper, 2 = Middle, 3 = Lower)", 1, 3, 3)
age = st.number_input("Age", 0, 100, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare Paid", 0.0, 600.0, 30.0)

# Prepare input for model
input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare]
})

# ---------------------------------------------------------
# Prediction Button
# ---------------------------------------------------------
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]

    st.subheader("🔍 Prediction Result:")
    if prediction == 1:
        st.success("🎉 The passenger **SURVIVED**!")
    else:
        st.error("❌ The passenger **DID NOT SURVIVE**.")
