import streamlit as st
import pandas as pd
import joblib
# ----------------------------
# Load Saved Scaler + KNN Model
# ----------------------------
scaler = joblib.load("Jawad-Titanic-main/scaler.pkl")
model = joblib.load("Jawad-Titanic-main/titanic.pkl")

st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details to predict **Survival (0 = No, 1 = Yes)**")

# ----------------------------
# Input Fields
# ----------------------------
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=50.0)

pclass = st.selectbox("Passenger Class", [1, 2, 3])
family = st.number_input("Family Size", min_value=0, max_value=11, value=1)

# One-hot encode Pclass inputs
Pclass_1 = 1 if pclass == 1 else 0
Pclass_2 = 1 if pclass == 2 else 0
Pclass_3 = 1 if pclass == 3 else 0

# ----------------------------
# Prepare input as a DF
# ----------------------------
input_df = pd.DataFrame([{
    "Age": age,
    "Fare": fare,
    "Pclass_1": Pclass_1,
    "Pclass_2": Pclass_2,
    "Pclass_3": Pclass_3,
    "Family_size": family
}])

# ----------------------------
# Predict
# ----------------------------
if st.button("Predict Survival"):
    # scale numerical features
    scaled_features = scaler.transform(input_df)

    # model prediction
    pred = model.predict(scaled_features)[0]
    prob = model.predict_proba(scaled_features)[0][1]

    st.subheader("🔍 Prediction Result:")
    if pred == 1:
        st.success("✅ Passenger is LIKELY to Survive!")
    else:
        st.error("❌ Passenger is NOT Likely to Survive")

    st.write(f"**Survival Probability:** {prob*100:.2f}%")
