import streamlit as st
import pandas as pd
import joblib

# Load trained model

model = joblib.load("Jawad-Fruit-main/rf_apples_oranges.pkl")

st.title("Apples vs Oranges Classifier")
st.write("Enter the weight and size of the fruit to predict whether it is Apple or Orange.")

weight = st.number_input("Weight (grams)", min_value=0.0, max_value=500.0, value=70.0)
size = st.number_input("Size", min_value=0.0, max_value=10.0, value=5.0)

if st.button("Predict"):
    input_df = pd.DataFrame([[weight, size]], columns=['Weight', 'Size'])
    prediction = model.predict(input_df)[0]
    fruit = "Apple" if prediction == 1 else "Orange"
    st.success(f"The predicted fruit is: {fruit}")
