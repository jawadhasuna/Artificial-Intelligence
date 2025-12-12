import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Custom Euclidean Distance
# ----------------------------
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# ----------------------------
# Custom KNN Classifier
# ----------------------------
class CustomKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X.values
        self.y_train = y.values

    def predict_one(self, x):
        distances = []
        for i in range(len(self.X_train)):
            dist = euclidean_distance(self.X_train[i], x)
            distances.append((dist, self.y_train[i]))
        distances.sort(key=lambda x: x[0])
        k_neighbors = distances[:self.k]
        labels = [label for _, label in k_neighbors]
        prediction = max(set(labels), key=labels.count)
        return prediction

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            # Ensure X is a DataFrame or convert to DataFrame
            if isinstance(X, pd.DataFrame):
                x_values = X.iloc[i].values
            else:
                x_values = X[i]
            predictions.append(self.predict_one(x_values))
        return np.array(predictions)

# ----------------------------
# Load the trained model
# ----------------------------
model = joblib.load("Jawad-Titanic-main/custom_knn_k20.pkl")  # <- your trained k=20 model

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🚢 Titanic Survival Prediction (Custom KNN)")

st.subheader("Enter Passenger Details:")

# Input features based on your training columns
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)
pclass_1 = st.checkbox("Pclass 1", value=False)
pclass_2 = st.checkbox("Pclass 2", value=False)
pclass_3 = st.checkbox("Pclass 3", value=True)
family_size = st.number_input("Family Size", min_value=0.0, max_value=10.0, value=1.0)

# Create input dataframe
input_data = pd.DataFrame({
    'Age': [age],
    'Fare': [fare],
    'Pclass_1': [1 if pclass_1 else 0],
    'Pclass_2': [1 if pclass_2 else 0],
    'Pclass_3': [1 if pclass_3 else 0],
    'Family_size': [family_size]
})

# Prediction button
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]

    st.subheader("🔍 Prediction Result:")
    if prediction == 1:
        st.success("Survived ✅")
    else:
        st.error("Did not Survive ❌")
