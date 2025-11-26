import streamlit as st
import numpy as np
import joblib

# Load trained model, scaler, and feature names
model = joblib.load("Jawad-Wisconsin-main/svm_rbf_model.pkl")
scaler = joblib.load("Jawad-Wisconsin-main/scaler.pkl")
selected_features = joblib.load("Jawad-Wisconsin-main/selected_features.pkl")

st.title("Breast Cancer Prediction App")
st.write("""
Enter the values for the top 5 features below. 
The app will predict whether the cancer is **Malignant** or **Benign**.
""")

# Optional: guide user with feature ranges (based on dataset)
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

# Create input fields dynamically for top features
inputs = []
for feature in selected_features:
    min_val, max_val = feature_ranges.get(feature, (0.0, 100.0))
    value = st.number_input(
        f"Enter {feature} (range: {min_val} - {max_val})", 
        min_value=float(min_val), 
        max_value=float(max_val), 
        step=0.01
    )
    inputs.append(value)

# Convert inputs to numpy array
user_input = np.array(inputs).reshape(1, -1)

# Scale the input
user_input_scaled = scaler.transform(user_input)

# Button to predict
if st.button("Predict"):
    prediction = model.predict(user_input_scaled)[0]
    confidence = model.decision_function(user_input_scaled)[0]  # higher = more confidence

    if prediction == 1:
        st.error(f"Prediction: Malignant (Cancerous)")
    else:
        st.success(f"Prediction: Benign (Non-cancerous)")
    
    st.write(f"Confidence Score: {confidence:.2f}")

# Show model accuracy (from your test set)
svm_acc = 0.9649  # replace with your actual accuracy

st.write(f"Model Accuracy: {svm_acc*100:.2f}%")
