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
