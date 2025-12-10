import streamlit as st
import pandas as pd
import joblib

# Load trained KNN pipeline
model = joblib.load("Jawad-Titanic-main/titanic_knn_pipeline.pkl")

st.title("🚢 Titanic Survival Prediction")

age = st.number_input("Age", min_value=0, max_value=100, value=25)
fare = st.number_input("Fare", min_value=0.0, max_value=1000.0, value=32.0)
pclass_1 = st.number_input("Pclass_1", min_value=0, max_value=1, value=0)
pclass_2 = st.number_input("Pclass_2", min_value=0, max_value=1, value=0)
pclass_3 = st.number_input("Pclass_3", min_value=0, max_value=1, value=1)
family_size = st.number_input("Family Size", min_value=0.0, max_value=20.0, value=1.0)

input_df = pd.DataFrame({
    'Age': [age],
    'Fare': [fare],
    'Pclass_1': [pclass_1],
    'Pclass_2': [pclass_2],
    'Pclass_3': [pclass_3],
    'Family_size': [family_size]
})

if st.button("Predict Survival"):
    pred = model.predict(input_df)
    prob = model.predict_proba(input_df)
    if pred[0] == 1:
        st.success(f"✅ Likely to Survive! Probability: {prob[0][1]*100:.2f}%")
    else:
        st.error(f"❌ Likely Not to Survive. Probability: {prob[0][0]*100:.2f}%")
