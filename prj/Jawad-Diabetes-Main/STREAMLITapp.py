import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Jawad's Diabetes Prediction App", layout="centered")
st.title("🩺 Jawad's Diabetes Prediction System")
st.write("Select a model, enter patient details, and predict diabetes risk.")

# -----------------------------
# LOAD DATA
# -----------------------------
# Replace with your dataset path if needed
df = pd.read_csv("Jawad-Diabetes-Main/diabetes.csv")

# -----------------------------
# FEATURES & TARGET
# -----------------------------
y = df['Outcome']
X = df[['Insulin','Glucose','BloodPressure','Age','BMI']]

# -----------------------------
# SCALING
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
xtrain, xtest, ytrain, ytest = train_test_split(
    X_scaled, y, test_size=0.15, random_state=0
)

# -----------------------------
# TRAIN MODELS
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', probability=True)
}

trained_models = {}
conf_matrices = {}

for name, model in models.items():
    model.fit(xtrain, ytrain)
    trained_models[name] = model
    conf_matrices[name] = confusion_matrix(ytest, model.predict(xtest))

# -----------------------------
# USER MODEL SELECTION
# -----------------------------
model_name = st.selectbox("Choose a Model", list(trained_models.keys()))
model = trained_models[model_name]

# -----------------------------
# USER INPUT
# -----------------------------
st.subheader("Enter Patient Details")

insulin = st.number_input("Insulin", 0, 900, 100)
glucose = st.number_input("Glucose", 0, 300, 120)
bp = st.number_input("Blood Pressure", 0, 200, 70)
age = st.number_input("Age", 1, 120, 35)
bmi = st.number_input("BMI", 10.0, 60.0, 30.0)

input_data = pd.DataFrame({
    'Insulin': [insulin],
    'Glucose': [glucose],
    'BloodPressure': [bp],
    'Age': [age],
    'BMI': [bmi]
})

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Diabetic")
    else:
        st.success("✅ Not Diabetic")

    st.write("### Prediction Probability")
    prob_df = pd.DataFrame({
        "Class": ["No Diabetes", "Diabetes"],
        "Probability": probability
    })

    # Probability Bar Chart
    fig, ax = plt.subplots()
    ax.bar(prob_df["Class"], prob_df["Probability"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title(f"{model_name} Prediction Probability")
    st.pyplot(fig)

    # -----------------------------
    # CONFUSION MATRIX
    # -----------------------------
    st.write("### Confusion Matrix (Test Data)")
    cm = conf_matrices[model_name]

    fig2, ax2 = plt.subplots()
    ax2.imshow(cm)
    ax2.set_title(f"{model_name} Confusion Matrix")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")

    for i in range(2):
        for j in range(2):
            ax2.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig2)
