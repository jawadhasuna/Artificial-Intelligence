# Heart Disease Binary Classification Analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, classification_report

# 1. Load and explore dataset
df = pd.read_csv("Heart_disease_cleveland_new.csv")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Summary statistics
print("\n=== Summary Statistics (mean, std) ===")
print(df.describe().T[['mean','std']])  # mean and std for all columns

# Correlation with target
corr_matrix = df.corr()
print("\n=== Correlation with target ===")
corr_target = corr_matrix["target"].sort_values(ascending=False)
print(corr_target)

# Check missing values
print("\n=== Missing Values ===")
print(df.isnull().sum())

# Fill missing values with median (if any)
df.fillna(df.median(), inplace=True)

# 2. Analyze combinations of two variables vs target
# Histograms by target (age, thalach)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.histplot(df[df['target']==0]['age'], kde=False, color='blue', label='No Disease', bins=15)
sns.histplot(df[df['target']==1]['age'], kde=False, color='red', label='Disease', bins=15, alpha=0.7)
plt.legend(); plt.title("Age distribution by Target")
plt.subplot(1,2,2)
sns.histplot(df[df['target']==0]['thalach'], kde=False, color='blue', label='No Disease', bins=15)
sns.histplot(df[df['target']==1]['thalach'], kde=False, color='red', label='Disease', bins=15, alpha=0.7)
plt.legend(); plt.title("Max HR (thalach) distribution by Target")
plt.tight_layout()
plt.show()

# Scatter plots of feature pairs colored by target
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.scatterplot(data=df, x='age', y='chol', hue='target', palette='coolwarm')
plt.title("Age vs Cholesterol (by Target)")
plt.subplot(1,2,2)
sns.scatterplot(data=df, x='thalach', y='oldpeak', hue='target', palette='coolwarm')
plt.title("MaxHR vs ST Dep (by Target)")
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

# 3. PCA (3 components) with 3D visualization
X = df.drop("target", axis=1)
X_std = (X - X.mean()) / X.std()  # standardize features
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_std)
explained = pca.explained_variance_ratio_
print("\nExplained variance ratio by 3 principal components:", explained)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], c=df['target'], cmap='coolwarm', edgecolor='k')
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
plt.title("3D PCA (colored by target)")
plt.colorbar(scatter, label='Target')
plt.show()

# 4. Logistic Regression from scratch (using 3 features)
features3 = ['ca','thal','oldpeak']  # choose 3 features for demonstration
X3 = df[features3].values
y = df['target'].values

# Stage 1: Fit linear regression (normal equation)
X3_aug = np.concatenate([np.ones((X3.shape[0],1)), X3], axis=1)  # add intercept term
w_lin = np.linalg.inv(X3_aug.T.dot(X3_aug)).dot(X3_aug.T).dot(y)
print("\nLinear regression coefficients (intercept + weights):", w_lin)

# Stage 2: Apply logistic (sigmoid) on linear output
z = X3_aug.dot(w_lin)
y_prob_scratch = 1 / (1 + np.exp(-z))
y_pred_scratch = (y_prob_scratch >= 0.5).astype(int)
print("Scratch model accuracy (on full data):", accuracy_score(y, y_pred_scratch))

# 5. Logistic Regression using scikit-learn (same 3 features)
X_train, X_test, y_train, y_test = train_test_split(X3, y, test_size=0.3, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred_sk = clf.predict(X_test)
print("\n=== scikit-learn Logistic Regression Metrics (3 features) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_sk))
print("Precision:", precision_score(y_test, y_pred_sk))
print("Recall:", recall_score(y_test, y_pred_sk))
print("F1-score:", f1_score(y_test, y_pred_sk))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_sk))
print("Classification Report:\n", classification_report(y_test, y_pred_sk))

# Compare coefficients (sklearn)
print("sklearn coefficients (intercept + weights):", np.concatenate([clf.intercept_, clf.coef_.flatten()]))

# 6. PCA for feature selection (top 5 features by first component)
pca_full = PCA(n_components=5)
pca_full.fit(X_std)
loadings = np.abs(pca_full.components_[0])
feature_importance = pd.Series(loadings, index=X.columns).sort_values(ascending=False)
top_features = feature_importance.index[:5].tolist()
print("\nTop 5 features by first principal component:", top_features)

# Logistic regression on top features (scratch and sklearn)
X_top = df[top_features].values
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_top, y, test_size=0.3, random_state=42)

# Scratch logistic (linear model then sigmoid) on top features
X2_aug = np.concatenate([np.ones((X_train2.shape[0],1)), X_train2], axis=1)
w2 = np.linalg.inv(X2_aug.T.dot(X2_aug)).dot(X2_aug.T).dot(y_train2)
X_test2_aug = np.concatenate([np.ones((X_test2.shape[0],1)), X_test2], axis=1)
z2_test = X_test2_aug.dot(w2)
y_prob2 = 1 / (1 + np.exp(-z2_test))
y_pred2 = (y_prob2 >= 0.5).astype(int)
print("\n=== Scratch Logistic Regression Metrics (top features) ===")
print("Accuracy:", accuracy_score(y_test2, y_pred2))
print("Precision:", precision_score(y_test2, y_pred2))
print("Recall:", recall_score(y_test2, y_pred2))
print("F1-score:", f1_score(y_test2, y_pred2))
print("Confusion Matrix:\n", confusion_matrix(y_test2, y_pred2))
print("Classification Report:\n", classification_report(y_test2, y_pred2))

# Sklearn logistic on top features
clf2 = LogisticRegression(max_iter=1000)
clf2.fit(X_train2, y_train2)
y_pred2_sk = clf2.predict(X_test2)
print("\n=== scikit-learn Logistic Regression (top features) ===")
print("Accuracy:", accuracy_score(y_test2, y_pred2_sk))
print("Precision:", precision_score(y_test2, y_pred2_sk))
print("Recall:", recall_score(y_test2, y_pred2_sk))
print("F1-score:", f1_score(y_test2, y_pred2_sk))
print("Confusion Matrix:\n", confusion_matrix(y_test2, y_pred2_sk))
print("Classification Report:\n", classification_report(y_test2, y_pred2_sk))

# 7. 2D Decision boundary example (scratch) using two features
# Use 'oldpeak' and 'thalach' for 2D logistic example
X2d = df[['oldpeak','thalach']].values
y2d = df['target'].values
X2d_aug = np.concatenate([np.ones((X2d.shape[0],1)), X2d], axis=1)
w2d = np.linalg.inv(X2d_aug.T.dot(X2d_aug)).dot(X2d_aug.T).dot(y2d)
# Decision boundary: w0 + w1*x + w2*y = 0 => y = -(w0 + w1*x)/w2
x_vals = np.linspace(X2d[:,0].min(), X2d[:,0].max(), 100)
y_vals = -(w2d[0] + w2d[1]*x_vals) / w2d[2]
plt.figure()
sns.scatterplot(x=df['oldpeak'], y=df['thalach'], hue=df['target'], palette='coolwarm', legend='brief')
plt.plot(x_vals, y_vals, color='black', label='Decision boundary')
plt.xlabel('Oldpeak'); plt.ylabel('Thalach (Max HR)')
plt.title('Decision Boundary (scratch logistic, 2 features)')
plt.legend()
plt.show()

# 8. ROC Curve and AUC (sklearn logistic with 3 features)
y_proba_test = clf.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_proba_test)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve (sklearn logistic)')
plt.legend(loc='lower right')
plt.show()
