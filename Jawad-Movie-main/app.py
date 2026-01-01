import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="IMDb Movie Clustering", layout="wide")

st.title("🎬 IMDb Movie Clustering Explorer")
st.markdown("Explore movie clusters using **unsupervised learning**")

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Jawad-Movie-main/merged_dataset.csv")
    return df

df = load_data()

# ------------------ PREPROCESSING ------------------
def convert_runtime(runtime):
    if pd.isna(runtime):
        return np.nan
    runtime = runtime.lower()
    hours, minutes = 0, 0
    if 'h' in runtime:
        parts = runtime.split('h')
        hours = int(parts[0])
        if 'min' in parts[1]:
            minutes = int(parts[1].replace('min','').strip())
    elif 'min' in runtime:
        minutes = int(runtime.replace('min','').strip())
    return hours * 60 + minutes

df['runtime_min'] = df['run_length'].apply(convert_runtime)

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Controls")

features = ['rating', 'runtime_min', 'num_raters', 'num_reviews']
x_feature = st.sidebar.selectbox("X-axis feature", features, index=0)
y_feature = st.sidebar.selectbox("Y-axis feature", features, index=1)

algo = st.sidebar.radio("Clustering Algorithm", ["K-Means", "DBSCAN"])
scale_data = st.sidebar.checkbox("Scale Features", value=False)

# ------------------ DATA PREP ------------------
X = df[[x_feature, y_feature]].dropna()

if scale_data:
    scaler = StandardScaler()
    X_values = scaler.fit_transform(X)
else:
    X_values = X.to_numpy()

# ------------------ CLUSTERING ------------------
if algo == "K-Means":
    k = st.sidebar.slider("Number of clusters (k)", 2, 6, 3)
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X_values)

else:
    eps = st.sidebar.slider("eps", 0.5, 20.0, 2.0)
    min_samples = st.sidebar.slider("min_samples", 2, 20, 5)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_values)

# ------------------ EVALUATION ------------------
st.subheader("📊 Clustering Evaluation")

try:
    if algo == "DBSCAN":
        mask = labels != -1
        sil = silhouette_score(X_values[mask], labels[mask])
        db = davies_bouldin_score(X_values[mask], labels[mask])
    else:
        sil = silhouette_score(X_values, labels)
        db = davies_bouldin_score(X_values, labels)

    st.metric("Silhouette Score", round(sil, 3))
    st.metric("Davies–Bouldin Index", round(db, 3))
except:
    st.warning("Not enough clusters to calculate evaluation metrics.")

# ------------------ VISUALIZATION ------------------
st.subheader("📈 Cluster Visualization")

fig, ax = plt.subplots(figsize=(7,5))
scatter = ax.scatter(
    X_values[:,0],
    X_values[:,1],
    c=labels,
    cmap='plasma'
)

ax.set_xlabel(x_feature)
ax.set_ylabel(y_feature)
ax.set_title(f"{algo} Clustering Result")
plt.colorbar(scatter)

st.pyplot(fig)

# ------------------ CLUSTER SUMMARY ------------------
st.subheader("📋 Cluster Summary")
summary = pd.DataFrame({"Cluster": labels}).value_counts().sort_index()
st.dataframe(summary)
