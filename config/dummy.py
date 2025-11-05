import numpy as np
import pandas as pd
from scipy.stats import entropy

# --------------------------------------------------------
# 1️⃣ Load and prep your data
# --------------------------------------------------------
# Expected columns: ['project_name','date','target','claim_amount', ...]
# date must be datetime64[ns]
df = pd.read_csv("claims_data.csv", parse_dates=["date"])

# Optional: ensure project_name is str
df["project_name"] = df["project_name"].astype(str)

# --------------------------------------------------------
# 2️⃣ Create basic time features
# --------------------------------------------------------
df["month_year"] = df["date"].dt.to_period("M").astype(str)
df["month_index"] = (df["date"].dt.year - df["date"].dt.year.min()) * 12 + df[
    "date"
].dt.month

latest_month_index = df["month_index"].max()


# --------------------------------------------------------
# 3️⃣ Compute Population Stability Index (PSI) per project
# --------------------------------------------------------
def psi(expected, actual, bins=10):
    """Calculate PSI between two numeric distributions."""
    if expected.nunique() < 2 or actual.nunique() < 2:
        return 0.0
    cut = np.linspace(
        min(expected.min(), actual.min()), max(expected.max(), actual.max()), bins + 1
    )
    e_perc = np.histogram(expected, bins=cut)[0] / len(expected)
    a_perc = np.histogram(actual, bins=cut)[0] / len(actual)
    e_perc = np.clip(e_perc, 1e-6, None)
    a_perc = np.clip(a_perc, 1e-6, None)
    return np.sum((a_perc - e_perc) * np.log(a_perc / e_perc))


# Choose baseline vs current window
baseline = df[df["date"] < "2025-01-01"]
current = df[df["date"] >= "2025-01-01"]

drift_scores = {}
for proj in df["project_name"].unique():
    base = baseline[baseline["project_name"] == proj]
    curr = current[current["project_name"] == proj]
    if len(base) > 0 and len(curr) > 0:
        score = psi(base["claim_amount"], curr["claim_amount"])
        drift_scores[proj] = score
    else:
        drift_scores[proj] = 0.0

drift_df = pd.DataFrame.from_dict(
    drift_scores, orient="index", columns=["psi"]
).reset_index()
drift_df.rename(columns={"index": "project_name"}, inplace=True)
print("\n=== Drift (PSI) by project ===")
print(drift_df.sort_values("psi", ascending=False))

# --------------------------------------------------------
# 4️⃣ Convert drift scores → drift weights
# --------------------------------------------------------
max_drift = max(drift_scores.values()) if max(drift_scores.values()) > 0 else 1.0
drift_df["drift_weight"] = 1.0 + drift_df["psi"] / max_drift  # e.g. 1–2× scale
df = df.merge(drift_df[["project_name", "drift_weight"]], on="project_name", how="left")

# --------------------------------------------------------
# 5️⃣ Per-project class weights (handle imbalance inside project)
# --------------------------------------------------------
class_stats = (
    df.groupby("project_name")["target"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .rename(columns={0: "nondup_ratio", 1: "dup_ratio"})
)
class_stats["pos_weight"] = class_stats["nondup_ratio"] / (
    class_stats["dup_ratio"] + 1e-6
)
df = df.merge(class_stats[["pos_weight"]], on="project_name", how="left")
df["class_weight"] = np.where(df["target"] == 1, df["pos_weight"], 1.0)

# --------------------------------------------------------
# 6️⃣ Time-decay weighting (favor recent data)
# --------------------------------------------------------
half_life_months = 6
lambda_ = np.log(2) / half_life_months
df["time_weight"] = np.exp(-lambda_ * (latest_month_index - df["month_index"]))

# --------------------------------------------------------
# 7️⃣ Combine all into one unified final weight
# --------------------------------------------------------
df["final_weight"] = df["class_weight"] * df["time_weight"] * df["drift_weight"]

# Optional sanity check
print("\n=== Average weight per project ===")
print(df.groupby("project_name")["final_weight"].mean().round(3))

# --------------------------------------------------------
# 8️⃣ Rolling-window features (within each project)
# --------------------------------------------------------
df = df.sort_values(["project_name", "date"])
df["avg_claim_amt_6m"] = df.groupby("project_name")["claim_amount"].transform(
    lambda s: s.rolling(window=6, min_periods=1).mean()
)
df["dup_rate_3m"] = df.groupby("project_name")["target"].transform(
    lambda s: s.rolling(window=3, min_periods=1).mean()
)

# --------------------------------------------------------
# 9️⃣ Now you can train CatBoost with df['final_weight']
# --------------------------------------------------------
from catboost import CatBoostClassifier, Pool

cat_features = ["project_name", "month_year"]

X = df.drop(columns=["target"])
y = df["target"]

pool = Pool(X, y, weight=df["final_weight"], cat_features=cat_features)

model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=8,
    eval_metric="Precision",
    random_seed=42,
    early_stopping_rounds=100,
)
model.fit(pool, verbose=200)
model.save_model("catboost_driftaware_model.cbm")

# --------------------------------------------------------
# 10️⃣ Optional: inspect top feature importances
# --------------------------------------------------------
fi = pd.DataFrame(
    {"feature": model.feature_names_, "importance": model.get_feature_importance()}
)
print("\n=== Top features ===")
print(fi.sort_values("importance", ascending=False).head(10))


#################
## on testing data

import numpy as np
import pandas as pd
from catboost import Pool

# --------------------------------------------------------
# 🧩 1️⃣ Inputs — you already have train_df and test_df
# --------------------------------------------------------
# train_df: data till August 2025 (already processed with final_weight)
# test_df:  data for Sept–Oct 2025, unprocessed yet
# Columns must include: ['project_name', 'date', 'target', ...]

# Example:
# train_df = df[df["date"] < "2025-09-01"].copy()
# test_df  = df[df["date"] >= "2025-09-01"].copy()

# --------------------------------------------------------
# 🧠 2️⃣ Extract mappings directly from training data
# --------------------------------------------------------
# Class weight map (avg per project)
class_map = train_df.groupby("project_name")["class_weight"].mean().to_dict()

# Drift weight map (avg per project)
drift_map = train_df.groupby("project_name")["drift_weight"].mean().to_dict()

# Compute latest month index from training data
latest_month_index = train_df["month_index"].max()

# Retrieve lambda from your time-decay formula
# (6-month half-life as used before)
lambda_ = np.log(2) / 6

print("\n✅ Mappings extracted from training data:")
print("class_map:", class_map)
print("drift_map:", drift_map)
print("latest_month_index:", latest_month_index)

# --------------------------------------------------------
# 🕒 3️⃣ Create month features for testing data
# --------------------------------------------------------
test_df["month_year"] = test_df["date"].dt.to_period("M").astype(str)
test_df["month_index"] = (
    test_df["date"].dt.year - train_df["date"].dt.year.min()
) * 12 + test_df["date"].dt.month

# --------------------------------------------------------
# ⚖️ 4️⃣ Apply mappings and recompute new time weights
# --------------------------------------------------------
# Class weights (from training)
test_df["class_weight"] = np.where(
    test_df["target"] == 1, test_df["project_name"].map(class_map), 1.0
)
test_df["class_weight"].fillna(1.0, inplace=True)

# Drift weights (from training)
test_df["drift_weight"] = test_df["project_name"].map(drift_map)
test_df["drift_weight"].fillna(1.0, inplace=True)

# Time-decay weights (fresh for new months)
test_df["time_weight"] = np.exp(
    -lambda_ * (latest_month_index - test_df["month_index"])
)
test_df["time_weight"] = np.clip(test_df["time_weight"], 0, 1)

# --------------------------------------------------------
# 🧮 5️⃣ Combine everything into final weight
# --------------------------------------------------------
test_df["final_weight"] = (
    test_df["class_weight"] * test_df["drift_weight"] * test_df["time_weight"]
)

# --------------------------------------------------------
# 🔍 6️⃣ Sanity check
# --------------------------------------------------------
print("\n=== TESTING DATA WEIGHT SUMMARY ===")
print(test_df.groupby("project_name")["final_weight"].describe().round(3))

# --------------------------------------------------------
# 🧠 7️⃣ Prepare CatBoost pool (optional)
# --------------------------------------------------------
cat_features = ["project_name", "month_year"]

# Replace with the same feature columns you used during training
feature_cols = [
    c for c in test_df.columns if c not in ["target", "date", "final_weight"]
]

test_pool = Pool(
    test_df[feature_cols],
    test_df["target"],
    weight=test_df["final_weight"],
    cat_features=cat_features,
)

# Now you can evaluate your trained CatBoost model on test_pool:
# preds = model.predict_proba(test_pool)[:, 1]
# auc = roc_auc_score(test_df["target"], preds, sample_weight=test_df["final_weight"])


## catboost

# =========================================================
#  CATBOOST + OPTUNA + VISUALIZATION PIPELINE
# =========================================================

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import shap
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    calibration_curve,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

# =========================================================
# 🧩 1️⃣ INPUTS
# =========================================================
# XTrainSelected, XTestSelected should already exist in your workspace.
# Must include: 'target', 'final_weight', and all feature columns.

target_col = "target"
weight_col = "final_weight"

# Automatically detect categorical features
cat_features = XTrainSelected.select_dtypes(
    include=["object", "category"]
).columns.tolist()
for col in [target_col, weight_col]:
    if col in cat_features:
        cat_features.remove(col)

# Define features
exclude_cols = [target_col, weight_col]
feature_cols = [c for c in XTrainSelected.columns if c not in exclude_cols]

print(f"✅ Detected {len(cat_features)} categorical features.")
print(f"✅ Using {len(feature_cols)} total features for training.")

# Create CatBoost Pools
train_pool = Pool(
    data=XTrainSelected[feature_cols],
    label=XTrainSelected[target_col],
    weight=XTrainSelected[weight_col],
    cat_features=cat_features,
)
test_pool = Pool(
    data=XTestSelected[feature_cols],
    label=XTestSelected[target_col],
    weight=XTestSelected[weight_col],
    cat_features=cat_features,
)


# =========================================================
# 🎯 2️⃣ OPTUNA OBJECTIVE
# =========================================================
def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 500, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10.0),
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
        ),
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": 42,
        "verbose": 0,
    }

    model = CatBoostClassifier(**params)
    model.fit(
        train_pool, eval_set=test_pool, early_stopping_rounds=100, use_best_model=True
    )
    preds = model.predict_proba(XTestSelected[feature_cols])[:, 1]
    auc = roc_auc_score(
        XTestSelected[target_col], preds, sample_weight=XTestSelected[weight_col]
    )
    return auc


# =========================================================
# ⚙️ 3️⃣ RUN OPTUNA TUNING
# =========================================================
study = optuna.create_study(direction="maximize", study_name="CatBoost_Tuning")
study.optimize(objective, n_trials=30, show_progress_bar=True)

print("\n🎯 Best trial parameters:")
for k, v in study.best_trial.params.items():
    print(f"  {k}: {v}")

# =========================================================
# 🧠 4️⃣ TRAIN FINAL MODEL
# =========================================================
best_params = study.best_trial.params
best_params.update(
    {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": 42,
        "verbose": 100,
    }
)

final_model = CatBoostClassifier(**best_params)
final_model.fit(
    train_pool,
    eval_set=test_pool,
    early_stopping_rounds=100,
    use_best_model=True,
    plot=True,
)

# =========================================================
# 📊 5️⃣ EVALUATION
# =========================================================
pred_probs = final_model.predict_proba(XTestSelected[feature_cols])[:, 1]
preds = (pred_probs >= 0.5).astype(int)
y_true = XTestSelected[target_col]
weights = XTestSelected[weight_col]

# Metrics
auc = roc_auc_score(y_true, pred_probs, sample_weight=weights)
f1 = f1_score(y_true, preds, sample_weight=weights)
precision = precision_score(y_true, preds, sample_weight=weights)
recall = recall_score(y_true, preds, sample_weight=weights)
accuracy = accuracy_score(y_true, preds, sample_weight=weights)
brier = brier_score_loss(y_true, pred_probs, sample_weight=weights)
logloss = log_loss(y_true, pred_probs, sample_weight=weights)
tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

print("\n=== FINAL MODEL PERFORMANCE ===")
print(f"AUC:           {auc:.4f}")
print(f"Accuracy:      {accuracy:.4f}")
print(f"F1 Score:      {f1:.4f}")
print(f"Precision:     {precision:.4f}")
print(f"Recall:        {recall:.4f}")
print(f"Brier Score:   {brier:.4f}")
print(f"Log Loss:      {logloss:.4f}")
print(f"False Negatives: {fn},  False Positives: {fp}")
print(f"True Positives:  {tp},  True Negatives: {tn}")

# =========================================================
# 📈 6️⃣ VISUALIZATIONS
# =========================================================

# --- Learning Curve ---
metrics = final_model.get_evals_result()
plt.figure(figsize=(8, 4))
plt.plot(metrics["learn"]["AUC"], label="Train AUC")
plt.plot(metrics["validation"]["AUC"], label="Validation AUC")
plt.xlabel("Iterations")
plt.ylabel("AUC")
plt.title("CatBoost Learning Curve")
plt.legend()
plt.show()

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# --- Calibration Curve ---
prob_true, prob_pred = calibration_curve(y_true, pred_probs, n_bins=10)
plt.plot(prob_pred, prob_true, marker="o")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("Predicted Probability")
plt.ylabel("True Probability")
plt.title("Calibration Curve")
plt.show()

# --- Per-Project Performance ---
if "project_name" in XTestSelected.columns:
    proj_perf = (
        XTestSelected.assign(pred=preds)
        .groupby("project_name")[["target", "pred"]]
        .agg({"target": "mean", "pred": "mean"})
        .rename(columns={"target": "ActualDupRate", "pred": "PredDupRate"})
    )
    proj_perf.plot(kind="bar", figsize=(10, 4))
    plt.title("Actual vs Predicted Duplicate Rate per Project")
    plt.ylabel("Duplicate Rate")
    plt.show()

# --- SHAP Summary Plot ---
print("\n🔍 Generating SHAP summary plot (may take a minute)...")
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(XTestSelected[feature_cols])
shap.summary_plot(shap_values, XTestSelected[feature_cols])

# --- False Positive vs True Positive Distribution Example ---
if "claim_amount" in XTestSelected.columns:
    fp = XTestSelected[(preds == 1) & (y_true == 0)]
    tp = XTestSelected[(preds == 1) & (y_true == 1)]
    plt.figure(figsize=(6, 3))
    sns.kdeplot(fp["claim_amount"], label="False Positives", shade=True)
    sns.kdeplot(tp["claim_amount"], label="True Positives", shade=True)
    plt.title("Claim Amount Distribution — FP vs TP")
    plt.legend()
    plt.show()

# =========================================================
# 💾 7️⃣ SAVE MODEL & IMPORTANCE
# =========================================================
final_model.save_model("catboost_best_model.cbm")
print("\n✅ Model saved as 'catboost_best_model.cbm'")

feat_imp = final_model.get_feature_importance(prettified=True)
feat_imp.to_csv("feature_importance.csv", index=False)
print("📊 Feature importance saved as 'feature_importance.csv'")
