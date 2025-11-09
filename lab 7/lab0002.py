# ===================================================================
# Full analysis & rich plotting for Heart Disease (7 features)
# - PCA (all numeric predictors) and many PCA visualizations
# - EDA: histograms, boxplots, pairplots, correlations
# - Selection justification plots showing why we picked the 7 features
# - Logistic regression: sklearn and custom GD (batch)
# - Exhaustive diagnostics: ROC, PR, CV-ROC, calibration, confusion matrices,
#   coefficient comparisons, GD trajectories, per-row probability visuals.
# - Saves many figures and CSV files for lab submission.
# ===================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, brier_score_loss
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid", context="talk")
RNG_SEED = 42
np.random.seed(RNG_SEED)

# -------------------------
# Config: paths & variables
# -------------------------
DATA_PATH = "Heart_disease_cleveland_new.csv"  # change if needed
OUT_DIR = "analysis_plots_and_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Full list of chosen features (first 4 from earlier + 3 PCA picks)
features7 = ['ca', 'thal', 'exang', 'thalach', 'cp', 'restecg', 'fbs']
target_col = 'target'

# -------------------------
# 0) Load data & sanity checks
# -------------------------
df = pd.read_csv(DATA_PATH)
# Ensure binary target (0 vs >0)
df[target_col] = (df[target_col] > 0).astype(int)

print("Shape:", df.shape)
print("Target distribution:\n", df[target_col].value_counts())
print("Are all 7 features present?", all([f in df.columns for f in features7]))
missing = [f for f in features7 if f not in df.columns]
if missing:
    raise ValueError(f"Missing features: {missing}")

# -------------------------
# 1) PCA on ALL numeric predictors (diagnostics)
# -------------------------
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != target_col]

# Standardize
X_all = df[num_cols].copy()
scaler_all = StandardScaler()
X_all_scaled = scaler_all.fit_transform(X_all)

# PCA
pca = PCA()
pcs = pca.fit_transform(X_all_scaled)
explained = pca.explained_variance_ratio_
cum_explained = np.cumsum(explained)
pc_names = [f"PC{i+1}" for i in range(len(explained))]

# Scree & cumulative (bar + line)
plt.figure(figsize=(10,4))
plt.bar(range(1, len(explained)+1), explained, alpha=0.7, label='PC explained var')
plt.plot(range(1, len(explained)+1), cum_explained, marker='o', color='orange', label='cumulative')
plt.xlabel('Principal component')
plt.ylabel('Explained variance ratio')
plt.title('PCA scree and cumulative explained variance (all numeric predictors)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pca_scree_cumulative.png"), dpi=150)
plt.show()

# Top features by cumulative abs loadings across first K PCs
K = min(6, len(pc_names))
loadings = pd.DataFrame(pca.components_.T, index=num_cols, columns=pc_names)
abs_load = loadings.abs()
var_contrib = abs_load.iloc[:, :K].sum(axis=1).sort_values(ascending=False)

# Heatmap of loadings for top variables across first K PCs
top_n = min(14, len(var_contrib))
vars_to_plot = var_contrib.head(top_n).index.tolist()
plt.figure(figsize=(12,6))
sns.heatmap(abs_load.loc[vars_to_plot, abs_load.columns[:K]].T, annot=True, fmt=".3f",
            cmap="Reds", linewidths=0.2)
plt.title(f"PCA absolute loadings for top {top_n} variables across first {K} PCs")
plt.xlabel("Variable")
plt.ylabel("Principal Component")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pca_loadings_heatmap_topvars.png"), dpi=150)
plt.show()

# Biplot for PC1 vs PC2 (samples colored by target) with variable vectors
pc1 = pcs[:,0]
pc2 = pcs[:,1]
plt.figure(figsize=(8,6))
scatter = plt.scatter(pc1, pc2, c=df[target_col], cmap='coolwarm', alpha=0.7, s=30)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Biplot: PC1 vs PC2 (points colored by target)")
# draw vectors for top 8 contributors to PC1/PC2
loading_mags = np.sqrt(loadings['PC1']**2 + loadings['PC2']**2)
top_idx = loading_mags.sort_values(ascending=False).head(8).index
scale = max(pc1.max()-pc1.min(), pc2.max()-pc2.min()) * 0.6
for var in top_idx:
    x = loadings.loc[var, 'PC1'] * scale
    y = loadings.loc[var, 'PC2'] * scale
    plt.arrow(0,0,x,y, head_width=0.05*scale, head_length=0.05*scale, color='k', alpha=0.7)
    plt.text(x*1.12, y*1.12, var, fontsize=9)
plt.colorbar(scatter, label='target')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pca_biplot_pc1_pc2.png"), dpi=150)
plt.show()

# Save PCA contribution ranking table
var_contrib.head(20).to_csv(os.path.join(OUT_DIR, "pca_var_contrib_top20.csv"))

# -------------------------
# 2) EDA for the 7 features (many plots)
# -------------------------
# Histograms + KDE + boxplots
plt.figure(figsize=(14,10))
for i, col in enumerate(features7):
    plt.subplot(4,2,i+1)
    sns.histplot(df[col].dropna(), kde=True, bins=20)
    plt.title(f"Histogram/KDE: {col}")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "histograms_7_features.png"), dpi=150)
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(data=df[features7], orient='h')
plt.title("Boxplots (7 features)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "boxplots_7_features.png"), dpi=150)
plt.show()

# Pairplot (may be heavy but useful) - color by target
sns.pairplot(df[features7 + [target_col]], hue=target_col, corner=True, plot_kws={'alpha':0.6, 's':20})
plt.suptitle("Pairplot for 7 features colored by target", y=1.02)
plt.savefig(os.path.join(OUT_DIR, "pairplot_7_features.png"), dpi=150)
plt.show()

# Correlation heatmap (Spearman)
corr7 = df[features7].corr(method='spearman')
plt.figure(figsize=(7,6))
sns.heatmap(corr7, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Spearman correlation (7 features)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "corr7_spearman.png"), dpi=150)
plt.show()

# Feature distributions by target (violin + swarm)
plt.figure(figsize=(14,10))
for i, col in enumerate(features7):
    plt.subplot(4,2,i+1)
    sns.violinplot(x=target_col, y=col, data=df, inner='quartile')
    plt.title(f"{col} distribution by target")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "violin_by_target_7.png"), dpi=150)
plt.show()

# -------------------------
# 3) Justification summary plot: bar of PCA contribution and correlation with target
# -------------------------
# Contribution bar (from var_contrib) for the 7 features
contrib7 = var_contrib.reindex(features7).fillna(0)
plt.figure(figsize=(8,4))
contrib7.sort_values().plot.barh()
plt.xlabel("Sum abs loadings (first K PCs)")
plt.title("PCA contribution (7 features)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pca_contrib_7.png"), dpi=150)
plt.show()

# Correlation of each feature with target (point-biserial / spearman)
corr_with_target = df[features7].corrwith(df[target_col], method='spearman').sort_values()
plt.figure(figsize=(8,4))
corr_with_target.plot.barh()
plt.title("Spearman correlation of each feature with target")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "corr_with_target_7.png"), dpi=150)
plt.show()

# -------------------------
# 4) Prepare data & scaling for modeling
# -------------------------
X7 = df[features7].astype(float).values
scaler7 = StandardScaler()
X7_scaled = scaler7.fit_transform(X7)
y = df[target_col].values

# Train/test split (stratified)
X_tr, X_te, y_tr, y_te, Xtr_orig, Xte_orig = train_test_split(
    X7_scaled, y, X7, test_size=0.30, random_state=RNG_SEED, stratify=y)

# -------------------------
# 5) Fit scikit-learn logistic regression
# -------------------------
sk_clf = LogisticRegression(max_iter=3000, solver='lbfgs', random_state=RNG_SEED)
sk_clf.fit(X_tr, y_tr)
prob_tr_sk = sk_clf.predict_proba(X_tr)[:,1]
prob_te_sk = sk_clf.predict_proba(X_te)[:,1]
pred_te_sk_05 = (prob_te_sk >= 0.5).astype(int)

# -------------------------
# 6) Fit custom batch GD logistic regression (training) with trajectories
# -------------------------
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
lr = 0.1
max_iters = 5000
tol = 1e-7
w = np.zeros(X_tr.shape[1], dtype=float)
b = 0.0
loss_hist = []
coef_hist = []
for it in range(max_iters):
    z = X_tr.dot(w) + b
    p = sigmoid(z)
    loss = -np.mean(y_tr * np.log(np.clip(p, 1e-12, 1-1e-12)) + (1-y_tr) * np.log(np.clip(1-p, 1e-12, 1-1e-12)))
    error = p - y_tr
    gw = X_tr.T.dot(error) / X_tr.shape[0]
    gb = np.mean(error)
    w -= lr * gw
    b -= lr * gb
    loss_hist.append(loss)
    coef_hist.append(np.concatenate(([b], w.copy())))
    if it > 20 and abs(loss_hist[-2] - loss_hist[-1]) < tol:
        break

coef_gd = w.copy()
intercept_gd = b
prob_tr_gd = sigmoid(X_tr.dot(coef_gd) + intercept_gd)
prob_te_gd = sigmoid(X_te.dot(coef_gd) + intercept_gd)
pred_te_gd_05 = (prob_te_gd >= 0.5).astype(int)

# -------------------------
# 7) Training diagnostics plots
# -------------------------
# Loss curve
plt.figure(figsize=(7,4))
plt.plot(loss_hist)
plt.xlabel("Iteration")
plt.ylabel("Log-loss (training)")
plt.title("GD training loss curve")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "gd_loss_curve.png"), dpi=150)
plt.show()

# GD coefficient trajectories (scaled space)
coef_hist_arr = np.array(coef_hist)
iters = np.arange(coef_hist_arr.shape[0])
plt.figure(figsize=(9,5))
plt.plot(iters, coef_hist_arr[:,0], label='intercept', linewidth=1.5)
for i, f in enumerate(features7):
    plt.plot(iters, coef_hist_arr[:, i+1], label=f'coef_{f}', linewidth=1)
plt.xlabel("Iteration")
plt.ylabel("Coefficient value (scaled)")
plt.title("GD coefficient trajectories during training")
plt.legend(bbox_to_anchor=(1.02,1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "gd_coefficient_trajectories.png"), dpi=150)
plt.show()

# -------------------------
# 8) Model evaluation: ROC, PR, calibration, confusion matrix and tables
# -------------------------
# Youden thresholds from training
def youden_threshold(y_true, probs):
    fpr, tpr, thr = roc_curve(y_true, probs)
    j = tpr - fpr
    idx = np.nanargmax(j)
    return thr[idx]

thr_sk_tr = youden_threshold(y_tr, prob_tr_sk)
thr_gd_tr = youden_threshold(y_tr, prob_tr_gd)

# Predictions at thresholds
pred_te_sk_opt = (prob_te_sk >= thr_sk_tr).astype(int)
pred_te_gd_opt = (prob_te_gd >= thr_gd_tr).astype(int)

# Metrics function
def compute_metrics(y_true, probs, preds):
    return {
        "accuracy": np.round((preds==y_true).mean(),4),
        "roc_auc": np.round(roc_auc_score(y_true, probs),4),
        "brier": np.round(brier_score_loss(y_true, probs),4),
        "avg_precision": np.round(average_precision_score(y_true, probs),4)
    }

metrics_sk_05 = compute_metrics(y_te, prob_te_sk, pred_te_sk_05)
metrics_sk_opt = compute_metrics(y_te, prob_te_sk, pred_te_sk_opt)
metrics_gd_05 = compute_metrics(y_te, prob_te_gd, pred_te_gd_05)
metrics_gd_opt = compute_metrics(y_te, prob_te_gd, pred_te_gd_opt)

metrics_df = pd.DataFrame([
    ["sklearn(0.5)", metrics_sk_05['accuracy'], metrics_sk_05['roc_auc'], metrics_sk_05['brier'], metrics_sk_05['avg_precision']],
    [f"sklearn(opt={thr_sk_tr:.3f})", metrics_sk_opt['accuracy'], metrics_sk_opt['roc_auc'], metrics_sk_opt['brier'], metrics_sk_opt['avg_precision']],
    ["gd(0.5)", metrics_gd_05['accuracy'], metrics_gd_05['roc_auc'], metrics_gd_05['brier'], metrics_gd_05['avg_precision']],
    [f"gd(opt={thr_gd_tr:.3f})", metrics_gd_opt['accuracy'], metrics_gd_opt['roc_auc'], metrics_gd_opt['brier'], metrics_gd_opt['avg_precision']],
], columns=['method','accuracy','roc_auc','brier','avg_precision'])
display(metrics_df)
metrics_df.to_csv(os.path.join(OUT_DIR, "test_metrics_table.csv"), index=False)

# ROC and PR curves (test)
fpr_sk, tpr_sk, _ = roc_curve(y_te, prob_te_sk)
fpr_gd, tpr_gd, _ = roc_curve(y_te, prob_te_gd)
plt.figure(figsize=(7,6))
plt.plot(fpr_sk, tpr_sk, label=f'Sklearn AUC={auc(fpr_sk,tpr_sk):.3f}')
plt.plot(fpr_gd, tpr_gd, label=f'GD AUC={auc(fpr_gd,tpr_gd):.3f}')
plt.plot([0,1],[0,1],'k--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Test ROC - Sklearn vs GD")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "roc_test_sk_vs_gd.png"), dpi=150)
plt.show()

prec_sk, rec_sk, _ = precision_recall_curve(y_te, prob_te_sk)
prec_gd, rec_gd, _ = precision_recall_curve(y_te, prob_te_gd)
plt.figure(figsize=(7,6))
plt.plot(rec_sk, prec_sk, label=f'Sklearn AP={average_precision_score(y_te,prob_te_sk):.3f}')
plt.plot(rec_gd, prec_gd, label=f'GD AP={average_precision_score(y_te,prob_te_gd):.3f}')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Test Precision-Recall")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pr_test_sk_vs_gd.png"), dpi=150)
plt.show()

# Calibration (reliability diagram)
prob_true_sk, prob_pred_sk = calibration_curve(y_te, prob_te_sk, n_bins=10)
prob_true_gd, prob_pred_gd = calibration_curve(y_te, prob_te_gd, n_bins=10)
plt.figure(figsize=(7,6))
plt.plot(prob_pred_sk, prob_true_sk, 's-', label=f'Sklearn (Brier={brier_score_loss(y_te,prob_te_sk):.3f})')
plt.plot(prob_pred_gd, prob_true_gd, 'o-', label=f'GD (Brier={brier_score_loss(y_te,prob_te_gd):.3f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Calibration Curve (test)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "calibration_test_sk_vs_gd.png"), dpi=150)
plt.show()

# Confusion matrix heatmaps
def show_cm(y_true, preds, title, fname):
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(title)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, fname), dpi=150); plt.show()

show_cm(y_te, pred_te_sk_05, "Sklearn Confusion (0.5)", "cm_sk_0.5.png")
show_cm(y_te, pred_te_sk_opt, f"Sklearn Confusion (opt={thr_sk_tr:.3f})", "cm_sk_opt.png")
show_cm(y_te, pred_te_gd_05, "GD Confusion (0.5)", "cm_gd_0.5.png")
show_cm(y_te, pred_te_gd_opt, f"GD Confusion (opt={thr_gd_tr:.3f})", "cm_gd_opt.png")

# Classification reports saved
pd.DataFrame(classification_report(y_te, pred_te_sk_05, output_dict=True)).T.to_csv(os.path.join(OUT_DIR, "class_report_sk_0.5.csv"))
pd.DataFrame(classification_report(y_te, pred_te_gd_05, output_dict=True)).T.to_csv(os.path.join(OUT_DIR, "class_report_gd_0.5.csv"))

# -------------------------
# 9) Coefficients: unstandardize and show equations
# -------------------------
def unstandardize_scaled(coef_scaled, intercept_scaled, means, scales):
    coef_orig = coef_scaled / scales
    intercept_orig = intercept_scaled - np.sum(coef_scaled * means / scales)
    return intercept_orig, coef_orig

means7 = scaler7.mean_
scales7 = scaler7.scale_
coef_sk_scaled = sk_clf.coef_.flatten(); intercept_sk_scaled = sk_clf.intercept_[0]
b0_sk, beta_sk = unstandardize_scaled(coef_sk_scaled, intercept_sk_scaled, means7, scales7)
b0_gd, beta_gd = unstandardize_scaled(coef_gd, intercept_gd, means7, scales7)

def linear_eq_text(b0, betas, feats):
    terms = " + ".join([f"({b:.6f})*{f}" for b,f in zip(betas, feats)])
    return f"y_linear = {b0:.6f} + " + terms

eq_sk_text = linear_eq_text(b0_sk, beta_sk, features7)
eq_gd_text = linear_eq_text(b0_gd, beta_gd, features7)

print("\nSKLEARN linear equation (original units):\n", eq_sk_text)
print("\nGD linear equation (original units):\n", eq_gd_text)

# Coefficient comparison plot
coef_comp = pd.DataFrame({
    'feature': features7,
    'sk_unstd': beta_sk,
    'gd_unstd': beta_gd
}).set_index('feature')
coef_comp.plot.bar(figsize=(9,4))
plt.title("Unstandardized coefficients: Sklearn vs GD")
plt.ylabel("Coefficient (original units)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "coef_comp_unstandardized.png"), dpi=150)
plt.show()

# -------------------------
# 10) Cross-validation diagnostics (CV-ROC curves with folds)
# -------------------------
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RNG_SEED)

# Sklearn CV predicted probabilities
sk_probas_cv = cross_val_predict(sk_clf, X7_scaled, y, cv=cv, method='predict_proba')[:,1]

# GD CV: train GD on each split to get test fold probabilities
def gd_train_return_probas(X_train, y_train, X_test, lr=0.1, iters=2000):
    wloc = np.zeros(X_train.shape[1]); bloc = 0.0
    for _ in range(iters):
        z = X_train.dot(wloc) + bloc
        p = sigmoid(z)
        e = p - y_train
        gw = X_train.T.dot(e) / X_train.shape[0]
        gb = np.mean(e)
        wloc -= lr * gw
        bloc -= lr * gb
    return sigmoid(X_test.dot(wloc) + bloc)

gd_probas_cv = np.zeros(len(y))
for train_idx, test_idx in cv.split(X7_scaled, y):
    Xtr_cv, Xte_cv = X7_scaled[train_idx], X7_scaled[test_idx]
    ytr_cv = y[train_idx]
    gd_probas_cv[test_idx] = gd_train_return_probas(Xtr_cv, ytr_cv, Xte_cv, lr=0.1, iters=1000)

# Plot CV ROC: aggregate fold ROC by interpolation
from scipy import interp
mean_fpr = np.linspace(0,1,101)
def plot_cv_roc(probas_cv, label, color, fname):
    tprs = []
    aucs = []
    plt.figure(figsize=(7,6))
    for train_idx, test_idx in cv.split(X7_scaled, y):
        # compute on the test fold with a model inside cv above (we already have probas_cv computed overall)
        # get probs and y for this fold
        probs_fold = probas_cv[test_idx]
        y_fold = y[test_idx]
        fpr, tpr, _ = roc_curve(y_fold, probs_fold)
        aucs.append(auc(fpr,tpr))
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        plt.plot(fpr, tpr, alpha=0.2)
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color=color, label=f"{label} Mean ROC (AUC={mean_auc:.3f} ± {std_auc:.3f})", lw=2)
    plt.plot([0,1],[0,1],'k--', alpha=0.6)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"Cross-validated ROC for {label}")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, fname), dpi=150); plt.show()

plot_cv_roc(sk_probas_cv, "Sklearn Logistic (CV)", "tab:blue", "cvroc_sk.png")
plot_cv_roc(gd_probas_cv, "GD Logistic (CV)", "tab:green", "cvroc_gd.png")

# -------------------------
# 11) Per-row outputs (refit on full data) and probability visualizations
# -------------------------
# Refit both on full dataset for final per-row outputs
sk_full = LogisticRegression(max_iter=3000, solver='lbfgs', random_state=RNG_SEED)
sk_full.fit(X7_scaled, y)
prob_sk_full = sk_full.predict_proba(X7_scaled)[:,1]

# Refit GD on full data
w_full = np.zeros(X7_scaled.shape[1]); b_full = 0.0
for _ in range(3000):
    z = X7_scaled.dot(w_full) + b_full
    p = sigmoid(z)
    e = p - y
    gw = X7_scaled.T.dot(e) / X7_scaled.shape[0]
    gb = np.mean(e)
    w_full -= 0.1 * gw
    b_full -= 0.1 * gb
prob_gd_full = sigmoid(X7_scaled.dot(w_full) + b_full)

# Save per-row outputs
per_row = df[features7].copy()
per_row['target'] = df[target_col]
per_row['prob_sk_full'] = prob_sk_full
per_row['prob_gd_full'] = prob_gd_full
per_row.to_csv(os.path.join(OUT_DIR, "per_row_probs_sk_gd_full.csv"), index=False)
print("Saved per_row_probs_sk_gd_full.csv")

# Probability scatter and difference histogram
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.scatter(range(len(prob_sk_full)), prob_sk_full, label='sk_prob', s=20)
plt.scatter(range(len(prob_gd_full)), prob_gd_full, label='gd_prob', s=20, alpha=0.7)
plt.hlines(0.5, 0, len(prob_sk_full), colors='k', linestyles='--', label='0.5')
plt.legend(); plt.title("Predicted probabilities (full-data refits)")
plt.subplot(1,2,2)
plt.hist(prob_sk_full - prob_gd_full, bins=30)
plt.title("Histogram of probability differences (sk - gd)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "probs_scatter_and_diff.png"), dpi=150)
plt.show()

# Heatmap of probabilities sorted by predicted probability (for visual)
order_idx = np.argsort(prob_sk_full)
plt.figure(figsize=(8,4))
plt.imshow(np.vstack([prob_sk_full[order_idx], prob_gd_full[order_idx]]), aspect='auto', cmap='RdYlBu')
plt.yticks([0,1], ['sk_prob','gd_prob'])
plt.colorbar(label='predicted probability')
plt.title("Heatmap of predicted probabilities (sorted by sklearn prob)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "probs_heatmap_sorted.png"), dpi=150)
plt.show()

# -------------------------
# 12) Save key summary artifacts for lab report
# -------------------------
# Save coefficients (unstandardized) from full refits
def unstd_full(coef_scaled, intercept_scaled, means, scales):
    return unstandardize_scaled(coef_scaled, intercept_scaled, means, scales)

coef_sk_full_scaled = sk_full.coef_.flatten(); intercept_sk_full_scaled = sk_full.intercept_[0]
b0_sk_full, beta_sk_full = unstandardize_scaled(coef_sk_full_scaled, intercept_sk_full_scaled, means7, scales7)
coef_gd_full_scaled = w_full.copy(); intercept_gd_full_scaled = b_full
b0_gd_full, beta_gd_full = unstandardize_scaled(coef_gd_full_scaled, intercept_gd_full_scaled, means7, scales7)

coef_table = pd.DataFrame({
    'feature': features7,
    'sk_coef_unstd': beta_sk_full,
    'gd_coef_unstd': beta_gd_full
}).set_index('feature')
coef_table.to_csv(os.path.join(OUT_DIR, "coefficients_unstandardized_fullrefit.csv"))

# Save equations
with open(os.path.join(OUT_DIR, "linear_equations_fullrefit.txt"), "w") as fh:
    fh.write("Sklearn full refit linear equation:\n")
    fh.write(linear_eq_text(b0_sk_full, beta_sk_full, features7) + "\n\n")
    fh.write("GD full refit linear equation:\n")
    fh.write(linear_eq_text(b0_gd_full, beta_gd_full, features7) + "\n")

# Save metrics table produced earlier
metrics_df.to_csv(os.path.join(OUT_DIR, "metrics_test_summary.csv"), index=False)

print("All figures and tables saved to:", OUT_DIR)
print("End of script.")
