# import warnings

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
# from scipy import stats
# from scipy.spatial.distance import jensenshannon
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_auc_score
# from sklearn.preprocessing import LabelEncoder

# warnings.filterwarnings("ignore")


# class DataDriftAnalyzer:
#     """
#     Comprehensive drift detection for categorical features across time periods.
#     Detects distribution changes using multiple statistical tests and visualizations.
#     """

#     def __init__(self, df, month_col):
#         """
#         Initialize the analyzer.

#         Parameters:
#         -----------
#         df : pd.DataFrame
#             Input dataframe with time series data
#         month_col : str
#             Column name containing month/date information
#         """
#         self.df = df.copy()
#         self.month_col = month_col
#         self.months = sorted(df[month_col].unique())
#         self.features = [col for col in df.columns if col != month_col]
#         self.results = {}

#     def calculate_entropy(self, distribution):
#         """Calculate Shannon entropy for a distribution."""
#         probs = distribution / distribution.sum()
#         probs = probs[probs > 0]
#         return -np.sum(probs * np.log2(probs))

#     def calculate_psi(self, expected, actual, buckets=10):
#         """
#         Calculate Population Stability Index (PSI).

#         PSI < 0.1: No significant change
#         0.1 <= PSI < 0.2: Moderate change
#         PSI >= 0.2: Significant change
#         """
#         if expected.dtype == "object" or actual.dtype == "object":
#             expected_counts = expected.value_counts(normalize=True, sort=False)
#             actual_counts = actual.value_counts(normalize=True, sort=False)

#             all_categories = set(expected_counts.index) | set(actual_counts.index)
#             expected_pct = pd.Series(
#                 {cat: expected_counts.get(cat, 0.0001) for cat in all_categories}
#             )
#             actual_pct = pd.Series(
#                 {cat: actual_counts.get(cat, 0.0001) for cat in all_categories}
#             )
#         else:
#             breakpoints = np.linspace(
#                 min(expected.min(), actual.min()),
#                 max(expected.max(), actual.max()),
#                 buckets + 1,
#             )
#             expected_pct = pd.cut(
#                 expected, bins=breakpoints, include_lowest=True
#             ).value_counts(normalize=True, sort=False)
#             actual_pct = pd.cut(
#                 actual, bins=breakpoints, include_lowest=True
#             ).value_counts(normalize=True, sort=False)
#             expected_pct = expected_pct.fillna(0.0001)
#             actual_pct = actual_pct.fillna(0.0001)

#         psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
#         return psi

#     def chi_square_test(self, month1, month2, feature):
#         """Perform chi-square test for categorical variables."""
#         data1 = self.df[self.df[self.month_col] == month1][feature]
#         data2 = self.df[self.df[self.month_col] == month2][feature]

#         all_categories = set(data1.unique()) | set(data2.unique())
#         counts1 = data1.value_counts()
#         counts2 = data2.value_counts()

#         observed = []
#         for cat in all_categories:
#             observed.append([counts1.get(cat, 0), counts2.get(cat, 0)])

#         observed = np.array(observed)

#         try:
#             chi2, p_value, dof, expected = stats.chi2_contingency(observed.T)
#             return chi2, p_value
#         except:
#             return 0, 1.0

#     def calculate_js_divergence(self, month1, month2, feature):
#         """Calculate Jensen-Shannon divergence between two distributions."""
#         data1 = self.df[self.df[self.month_col] == month1][feature]
#         data2 = self.df[self.df[self.month_col] == month2][feature]

#         all_categories = sorted(set(data1.unique()) | set(data2.unique()))

#         counts1 = data1.value_counts(normalize=True)
#         counts2 = data2.value_counts(normalize=True)

#         p = np.array([counts1.get(cat, 1e-10) for cat in all_categories])
#         q = np.array([counts2.get(cat, 1e-10) for cat in all_categories])

#         p = p / p.sum()
#         q = q / q.sum()

#         return jensenshannon(p, q) ** 2

#     def model_based_drift_detection(self, month1, month2, feature):
#         """
#         Train a classifier to distinguish between two time periods.
#         Higher AUC indicates stronger drift.

#         AUC ~ 0.5: No drift
#         AUC > 0.6: Mild drift
#         AUC > 0.75: Strong drift
#         """
#         data1 = self.df[self.df[self.month_col] == month1][[feature]].copy()
#         data2 = self.df[self.df[self.month_col] == month2][[feature]].copy()

#         data1["target"] = 0
#         data2["target"] = 1

#         combined = pd.concat([data1, data2], ignore_index=True)

#         le = LabelEncoder()
#         combined[feature + "_encoded"] = le.fit_transform(combined[feature].astype(str))

#         X = combined[[feature + "_encoded"]].values
#         y = combined["target"].values

#         try:
#             model = LogisticRegression(random_state=42, max_iter=1000)
#             model.fit(X, y)
#             y_pred = model.predict_proba(X)[:, 1]
#             auc = roc_auc_score(y, y_pred)
#             return auc
#         except:
#             return 0.5

#     def analyze_feature(self, feature):
#         """Perform comprehensive drift analysis for a single feature."""
#         print(f"\n{'='*60}")
#         print(f"Analyzing feature: {feature}")
#         print(f"{'='*60}")

#         results = {
#             "feature": feature,
#             "monthly_distributions": {},
#             "monthly_entropy": {},
#             "drift_metrics": [],
#             "top_categories": None,
#         }

#         for month in self.months:
#             data = self.df[self.df[self.month_col] == month][feature]
#             dist = data.value_counts()
#             results["monthly_distributions"][month] = dist
#             results["monthly_entropy"][month] = self.calculate_entropy(dist)

#         all_values = self.df[feature].value_counts()
#         results["top_categories"] = all_values.head(5).index.tolist()

#         for i in range(len(self.months) - 1):
#             month1, month2 = self.months[i], self.months[i + 1]

#             data1 = self.df[self.df[self.month_col] == month1][feature]
#             data2 = self.df[self.df[self.month_col] == month2][feature]

#             chi2, p_value = self.chi_square_test(month1, month2, feature)
#             psi = self.calculate_psi(data1, data2)
#             jsd = self.calculate_js_divergence(month1, month2, feature)
#             auc = self.model_based_drift_detection(month1, month2, feature)

#             drift_detected = psi > 0.1 or jsd > 0.1 or auc > 0.6

#             metrics = {
#                 "comparison": f"{month1} -> {month2}",
#                 "chi_square": chi2,
#                 "p_value": p_value,
#                 "psi": psi,
#                 "js_divergence": jsd,
#                 "model_auc": auc,
#                 "drift_detected": drift_detected,
#             }

#             results["drift_metrics"].append(metrics)

#             print(f"\n{month1} -> {month2}:")
#             print(f"  Chi-square: {chi2:.2f} (p={p_value:.4f})")
#             print(f"  PSI: {psi:.4f} {'DRIFT DETECTED' if psi > 0.1 else 'Stable'}")
#             print(
#                 f"  JS Divergence: {jsd:.4f} {'DRIFT DETECTED' if jsd > 0.1 else 'Stable'}"
#             )
#             print(
#                 f"  Model AUC: {auc:.4f} {'DRIFT DETECTED' if auc > 0.6 else 'Stable'}"
#             )
#             print(
#                 f"  Overall: {'DRIFT DETECTED' if drift_detected else 'No significant drift'}"
#             )

#         return results

#     def analyze_all_features(self):
#         """Analyze drift for all features."""
#         print("\n" + "=" * 60)
#         print("COMPREHENSIVE DATA DRIFT ANALYSIS")
#         print("=" * 60)
#         print(f"Dataset: {len(self.df)} rows, {len(self.features)} features")
#         print(f"Time periods: {len(self.months)} months")
#         print(f"Months: {', '.join(map(str, self.months))}")

#         for feature in self.features:
#             self.results[feature] = self.analyze_feature(feature)

#         self._generate_summary()

#     def _generate_summary(self):
#         """Generate overall drift summary."""
#         print("\n" + "=" * 60)
#         print("DRIFT SUMMARY")
#         print("=" * 60)

#         summary = []
#         for feature, result in self.results.items():
#             metrics = result["drift_metrics"]
#             avg_psi = np.mean([m["psi"] for m in metrics])
#             max_psi = np.max([m["psi"] for m in metrics])
#             avg_jsd = np.mean([m["js_divergence"] for m in metrics])
#             avg_auc = np.mean([m["model_auc"] for m in metrics])
#             drift_rate = sum([m["drift_detected"] for m in metrics]) / len(metrics)

#             summary.append(
#                 {
#                     "Feature": feature,
#                     "Avg PSI": avg_psi,
#                     "Max PSI": max_psi,
#                     "Avg JSD": avg_jsd,
#                     "Avg AUC": avg_auc,
#                     "Drift Rate": drift_rate,
#                     "Status": (
#                         "HIGH DRIFT"
#                         if avg_psi > 0.2
#                         else ("MODERATE" if avg_psi > 0.1 else "STABLE")
#                     ),
#                 }
#             )

#         summary_df = pd.DataFrame(summary).sort_values("Avg PSI", ascending=False)
#         print("\n", summary_df.to_string(index=False))

#         high_drift_features = summary_df[summary_df["Avg PSI"] > 0.2][
#             "Feature"
#         ].tolist()
#         moderate_drift_features = summary_df[
#             (summary_df["Avg PSI"] > 0.1) & (summary_df["Avg PSI"] <= 0.2)
#         ]["Feature"].tolist()

#         print("\n" + "=" * 60)
#         print("CONCLUSION")
#         print("=" * 60)

#         if high_drift_features:
#             print(f"\nHIGH DRIFT detected in {len(high_drift_features)} feature(s):")
#             for f in high_drift_features:
#                 print(f"   - {f}")
#             print(
#                 "\n   These features show significant distribution changes over time."
#             )
#             print(
#                 "   The new data is NOT from the same distribution as earlier periods."
#             )

#         if moderate_drift_features:
#             print(
#                 f"\nMODERATE DRIFT detected in {len(moderate_drift_features)} feature(s):"
#             )
#             for f in moderate_drift_features:
#                 print(f"   - {f}")
#             print("\n   These features show noticeable distribution shifts.")

#         stable_features = (
#             len(summary_df) - len(high_drift_features) - len(moderate_drift_features)
#         )
#         if stable_features > 0:
#             print(f"\n{stable_features} feature(s) remain stable.")

#         print("\n" + "=" * 60)
#         print("RECOMMENDATIONS")
#         print("=" * 60)
#         print("1. Model retraining recommended for high-drift features")
#         print("2. Implement continuous monitoring for moderate-drift features")
#         print("3. Investigate root causes of distribution shifts")
#         print("4. Consider adaptive sampling or feature engineering")
#         print("5. Update training data to include recent distributions")

#         return summary_df

#     def plot_drift_summary(self, figsize=(14, 10)):
#         """Create comprehensive visualization of drift analysis."""
#         if not self.results:
#             print("Run analyze_all_features() first!")
#             return

#         summary = []
#         for feature, result in self.results.items():
#             metrics = result["drift_metrics"]
#             avg_psi = np.mean([m["psi"] for m in metrics])
#             summary.append({"Feature": feature, "Avg PSI": avg_psi})
#         summary_df = pd.DataFrame(summary).sort_values("Avg PSI", ascending=False)

#         fig = plt.figure(figsize=figsize)
#         gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

#         ax1 = fig.add_subplot(gs[0, :])
#         colors = [
#             "#ef4444" if x > 0.2 else "#f59e0b" if x > 0.1 else "#10b981"
#             for x in summary_df["Avg PSI"]
#         ]
#         ax1.barh(summary_df["Feature"], summary_df["Avg PSI"], color=colors)
#         ax1.axvline(
#             0.1, color="orange", linestyle="--", alpha=0.7, label="Moderate threshold"
#         )
#         ax1.axvline(0.2, color="red", linestyle="--", alpha=0.7, label="High threshold")
#         ax1.set_xlabel("Average PSI", fontsize=12, fontweight="bold")
#         ax1.set_title(
#             "Population Stability Index (PSI) by Feature",
#             fontsize=14,
#             fontweight="bold",
#         )
#         ax1.legend()
#         ax1.grid(axis="x", alpha=0.3)

#         top_feature = summary_df.iloc[0]["Feature"]
#         entropy_data = self.results[top_feature]["monthly_entropy"]
#         ax2 = fig.add_subplot(gs[1, 0])
#         months_list = list(entropy_data.keys())
#         entropy_values = list(entropy_data.values())
#         ax2.plot(
#             months_list,
#             entropy_values,
#             marker="o",
#             linewidth=2,
#             markersize=8,
#             color="#3b82f6",
#         )
#         ax2.set_xlabel("Month", fontsize=11, fontweight="bold")
#         ax2.set_ylabel("Entropy", fontsize=11, fontweight="bold")
#         ax2.set_title(
#             f"Entropy Evolution: {top_feature}", fontsize=12, fontweight="bold"
#         )
#         ax2.grid(alpha=0.3)
#         ax2.tick_params(axis="x", rotation=45)

#         ax3 = fig.add_subplot(gs[1, 1])
#         top_cats = self.results[top_feature]["top_categories"][:3]
#         for cat in top_cats:
#             percentages = []
#             for month in self.months:
#                 dist = self.results[top_feature]["monthly_distributions"][month]
#                 pct = (dist.get(cat, 0) / dist.sum()) * 100
#                 percentages.append(pct)
#             ax3.plot(self.months, percentages, marker="o", label=str(cat), linewidth=2)
#         ax3.set_xlabel("Month", fontsize=11, fontweight="bold")
#         ax3.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
#         ax3.set_title(
#             f"Top Categories Trend: {top_feature}", fontsize=12, fontweight="bold"
#         )
#         ax3.legend()
#         ax3.grid(alpha=0.3)
#         ax3.tick_params(axis="x", rotation=45)

#         ax4 = fig.add_subplot(gs[2, :])
#         auc_data = []
#         features_list = []
#         for feature, result in self.results.items():
#             aucs = [m["model_auc"] for m in result["drift_metrics"]]
#             auc_data.append(aucs)
#             features_list.append(feature)

#         comparisons = [
#             m["comparison"] for m in self.results[top_feature]["drift_metrics"]
#         ]
#         auc_df = pd.DataFrame(auc_data, columns=comparisons, index=features_list)

#         sns.heatmap(
#             auc_df,
#             annot=True,
#             fmt=".3f",
#             cmap="RdYlGn_r",
#             center=0.6,
#             ax=ax4,
#             cbar_kws={"label": "AUC Score"},
#             vmin=0.5,
#             vmax=0.9,
#         )
#         ax4.set_title(
#             "Model-Based Drift Detection (AUC per Month Pair)",
#             fontsize=12,
#             fontweight="bold",
#         )
#         ax4.set_xlabel("Month Comparison", fontsize=11, fontweight="bold")
#         ax4.set_ylabel("Feature", fontsize=11, fontweight="bold")

#         plt.suptitle(
#             "Comprehensive Data Drift Analysis Dashboard",
#             fontsize=16,
#             fontweight="bold",
#             y=0.995,
#         )

#         plt.tight_layout()
#         plt.savefig("drift_analysis_dashboard.png", dpi=300, bbox_inches="tight")
#         print("\nDashboard saved as 'drift_analysis_dashboard.png'")
#         plt.show()

#     def plot_feature_details(self, feature, figsize=(14, 8)):
#         """Create detailed visualization for a specific feature."""
#         if feature not in self.results:
#             print(f"Feature '{feature}' not found. Run analyze_all_features() first!")
#             return

#         result = self.results[feature]

#         fig, axes = plt.subplots(2, 2, figsize=figsize)
#         fig.suptitle(
#             f"Detailed Drift Analysis: {feature}", fontsize=16, fontweight="bold"
#         )

#         entropy_data = result["monthly_entropy"]
#         axes[0, 0].plot(
#             list(entropy_data.keys()),
#             list(entropy_data.values()),
#             marker="o",
#             linewidth=2,
#             markersize=8,
#             color="#3b82f6",
#         )
#         axes[0, 0].set_title("Entropy Evolution", fontweight="bold")
#         axes[0, 0].set_xlabel("Month", fontweight="bold")
#         axes[0, 0].set_ylabel("Entropy", fontweight="bold")
#         axes[0, 0].grid(alpha=0.3)
#         axes[0, 0].tick_params(axis="x", rotation=45)

#         metrics = result["drift_metrics"]
#         comparisons = [m["comparison"] for m in metrics]
#         psi_values = [m["psi"] for m in metrics]
#         colors = [
#             "#ef4444" if x > 0.2 else "#f59e0b" if x > 0.1 else "#10b981"
#             for x in psi_values
#         ]
#         axes[0, 1].bar(range(len(comparisons)), psi_values, color=colors)
#         axes[0, 1].axhline(0.1, color="orange", linestyle="--", alpha=0.7)
#         axes[0, 1].axhline(0.2, color="red", linestyle="--", alpha=0.7)
#         axes[0, 1].set_title("PSI Values (Month-to-Month)", fontweight="bold")
#         axes[0, 1].set_xlabel("Comparison", fontweight="bold")
#         axes[0, 1].set_ylabel("PSI", fontweight="bold")
#         axes[0, 1].set_xticks(range(len(comparisons)))
#         axes[0, 1].set_xticklabels(comparisons, rotation=45, ha="right")
#         axes[0, 1].grid(axis="y", alpha=0.3)

#         top_cats = result["top_categories"][:5]
#         for cat in top_cats:
#             percentages = []
#             for month in self.months:
#                 dist = result["monthly_distributions"][month]
#                 pct = (dist.get(cat, 0) / dist.sum()) * 100
#                 percentages.append(pct)
#             axes[1, 0].plot(
#                 self.months, percentages, marker="o", label=str(cat), linewidth=2
#             )
#         axes[1, 0].set_title("Top 5 Categories Frequency", fontweight="bold")
#         axes[1, 0].set_xlabel("Month", fontweight="bold")
#         axes[1, 0].set_ylabel("Percentage (%)", fontweight="bold")
#         axes[1, 0].legend(fontsize=8)
#         axes[1, 0].grid(alpha=0.3)
#         axes[1, 0].tick_params(axis="x", rotation=45)

#         jsd_values = [m["js_divergence"] for m in metrics]
#         auc_values = [m["model_auc"] for m in metrics]

#         x = np.arange(len(comparisons))
#         width = 0.35

#         axes[1, 1].bar(
#             x - width / 2, jsd_values, width, label="JS Divergence", color="#8b5cf6"
#         )
#         axes[1, 1].bar(
#             x + width / 2,
#             [a - 0.5 for a in auc_values],
#             width,
#             label="AUC - 0.5",
#             color="#ec4899",
#         )
#         axes[1, 1].set_title("Drift Metrics Comparison", fontweight="bold")
#         axes[1, 1].set_xlabel("Comparison", fontweight="bold")
#         axes[1, 1].set_ylabel("Value", fontweight="bold")
#         axes[1, 1].set_xticks(x)
#         axes[1, 1].set_xticklabels(comparisons, rotation=45, ha="right")
#         axes[1, 1].legend()
#         axes[1, 1].grid(axis="y", alpha=0.3)

#         plt.tight_layout()
#         plt.savefig(f"drift_analysis_{feature}.png", dpi=300, bbox_inches="tight")
#         print(f"\nFeature analysis saved as 'drift_analysis_{feature}.png'")
#         plt.show()


# # Example usage
# if __name__ == "__main__":
#     np.random.seed(42)

#     months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
#     data = []

#     for i, month in enumerate(months):
#         n_samples = 1000

#         cat_a_probs = [0.4, 0.35, 0.30, 0.25, 0.20, 0.15]
#         categories_a = np.random.choice(
#             ["X", "Y", "Z"],
#             n_samples,
#             p=[cat_a_probs[i], 0.3, 1 - cat_a_probs[i] - 0.3],
#         )

#         if i < 2:
#             categories_b = np.random.choice(
#                 ["A", "B", "C"], n_samples, p=[0.6, 0.3, 0.1]
#             )
#         else:
#             categories_b = np.random.choice(
#                 ["A", "B", "C"], n_samples, p=[0.2, 0.5, 0.3]
#             )

#         categories_c = np.random.choice(["P", "Q", "R"], n_samples, p=[0.5, 0.3, 0.2])

#         for j in range(n_samples):
#             data.append(
#                 {
#                     "month": month,
#                     "feature_gradual_drift": categories_a[j],
#                     "feature_sudden_drift": categories_b[j],
#                     "feature_stable": categories_c[j],
#                 }
#             )

#     df = pd.DataFrame(data)

#     print("\nSample Dataset Created:")
#     print(df.head(10))
#     print(f"\nShape: {df.shape}")

#     analyzer = DataDriftAnalyzer(df, month_col="month")
#     analyzer.analyze_all_features()

#     analyzer.plot_drift_summary()

#     analyzer.plot_feature_details("feature_gradual_drift")

#     print("\nAnalysis complete!")
#     print("\n" + "=" * 60)
#     print("INTERPRETATION:")
#     print("=" * 60)
#     print(
#         """
# This analysis proves data drift exists when:
# 1. PSI > 0.1 indicates distribution changes
# 2. Category frequencies shift significantly over time
# 3. Entropy increases (more randomness/diversity)
# 4. New categories appear or existing ones disappear
# 5. A simple model can distinguish between time periods (AUC > 0.6)

# When drift is detected, it means:
# - New data is NOT from the same distribution
# - Model performance may degrade
# - Retraining or monitoring is required
# - Feature relationships may have changed
#     """
#     )

# ##=============================================================

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
# from scipy import stats


# def drift_deep_dive(df, month_col, feature, month1, month2, top_n=15):
#     """
#     Deep dive into WHY a specific feature is drifting.
#     Shows which categories are changing and by how much.

#     Parameters:
#     -----------
#     df : DataFrame with your data
#     month_col : name of your month column
#     feature : the drifting feature you want to investigate
#     month1, month2 : two months to compare
#     top_n : number of categories to show
#     """

#     # Get data for both months
#     df1 = df[df[month_col] == month1]
#     df2 = df[df[month_col] == month2]

#     # Calculate distributions
#     dist1 = df1[feature].value_counts(normalize=True)
#     dist2 = df2[feature].value_counts(normalize=True)

#     # Combine and analyze
#     dist_df = pd.concat([dist1, dist2], axis=1, keys=[month1, month2]).fillna(0)
#     dist_df["change"] = dist_df[month2] - dist_df[month1]
#     dist_df["pct_change"] = (
#         (dist_df[month2] - dist_df[month1]) / (dist_df[month1] + 1e-10)
#     ) * 100
#     dist_df["abs_change"] = dist_df["change"].abs()

#     # Classify drift types
#     dist_df["drift_type"] = "Stable"
#     dist_df.loc[dist_df["change"] > 0.02, "drift_type"] = "Increasing"
#     dist_df.loc[dist_df["change"] < -0.02, "drift_type"] = "Decreasing"
#     dist_df.loc[(dist_df[month1] == 0) & (dist_df[month2] > 0), "drift_type"] = "NEW"
#     dist_df.loc[(dist_df[month1] > 0) & (dist_df[month2] == 0), "drift_type"] = (
#         "DISAPPEARED"
#     )

#     dist_df = dist_df.sort_values("abs_change", ascending=False)

#     # Print Analysis
#     print("=" * 80)
#     print(f"WHY IS '{feature}' DRIFTING?")
#     print(f"{month1} (n={len(df1):,}) → {month2} (n={len(df2):,})")
#     print("=" * 80)

#     # Overall metrics
#     tvd = 0.5 * dist_df["abs_change"].sum()
#     print(f"\nTotal Variation Distance: {tvd:.4f}")
#     if tvd > 0.2:
#         print("  → SEVERE drift (distribution fundamentally changed)")
#     elif tvd > 0.1:
#         print("  → MODERATE drift (noticeable shift)")
#     else:
#         print("  → MILD drift (minor changes)")

#     # What changed?
#     new_cats = dist_df[dist_df["drift_type"] == "NEW"]
#     disappeared = dist_df[dist_df["drift_type"] == "DISAPPEARED"]
#     increasing = dist_df[dist_df["drift_type"] == "Increasing"]
#     decreasing = dist_df[dist_df["drift_type"] == "Decreasing"]

#     print(f"\n📊 BREAKDOWN:")
#     print(f"   {len(new_cats):3d} NEW categories appeared")
#     print(f"   {len(disappeared):3d} categories DISAPPEARED")
#     print(f"   {len(increasing):3d} categories INCREASING")
#     print(f"   {len(decreasing):3d} categories DECREASING")

#     # Top changers
#     print(f"\n🔥 TOP {min(10, len(dist_df))} BIGGEST CHANGES:")
#     print("-" * 80)
#     for i, (cat, row) in enumerate(dist_df.head(10).iterrows(), 1):
#         m1_pct = row[month1] * 100
#         m2_pct = row[month2] * 100
#         change = row["change"] * 100
#         dtype = row["drift_type"]

#         arrow = "📈" if change > 0 else "📉" if change < 0 else "➡️"
#         print(f"{i:2d}. {cat[:40]:40s} {arrow} {dtype:12s}")
#         print(
#             f"    {month1}: {m1_pct:6.2f}%  →  {month2}: {m2_pct:6.2f}%  (change: {change:+.2f}%)"
#         )

#     # Category details by type
#     if len(new_cats) > 0:
#         print(f"\n✨ NEW CATEGORIES (why they appeared):")
#         for cat in new_cats.head(5).index:
#             pct = new_cats.loc[cat, month2] * 100
#             print(f"   • {cat}: {pct:.2f}% of {month2} data")

#     if len(disappeared) > 0:
#         print(f"\n❌ DISAPPEARED CATEGORIES (why they're gone):")
#         for cat in disappeared.head(5).index:
#             pct = disappeared.loc[cat, month1] * 100
#             print(f"   • {cat}: was {pct:.2f}% in {month1}")

#     if len(increasing) > 0:
#         print(f"\n📈 GROWING CATEGORIES (gaining share):")
#         for cat in increasing.head(5).index:
#             m1 = increasing.loc[cat, month1] * 100
#             m2 = increasing.loc[cat, month2] * 100
#             print(f"   • {cat}: {m1:.2f}% → {m2:.2f}% (+{m2-m1:.2f}%)")

#     if len(decreasing) > 0:
#         print(f"\n📉 SHRINKING CATEGORIES (losing share):")
#         for cat in decreasing.head(5).index:
#             m1 = decreasing.loc[cat, month1] * 100
#             m2 = decreasing.loc[cat, month2] * 100
#             print(f"   • {cat}: {m1:.2f}% → {m2:.2f}% ({m2-m1:.2f}%)")

#     # Create visualizations
#     fig = plt.figure(figsize=(18, 10))
#     gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

#     # 1. Before/After comparison
#     ax1 = fig.add_subplot(gs[0, :2])
#     top_cats = dist_df.head(top_n).index
#     x = np.arange(len(top_cats))
#     width = 0.35

#     ax1.bar(
#         x - width / 2,
#         dist_df.loc[top_cats, month1] * 100,
#         width,
#         label=month1,
#         alpha=0.8,
#         color="#3b82f6",
#     )
#     ax1.bar(
#         x + width / 2,
#         dist_df.loc[top_cats, month2] * 100,
#         width,
#         label=month2,
#         alpha=0.8,
#         color="#ef4444",
#     )
#     ax1.set_ylabel("Percentage (%)", fontsize=12, fontweight="bold")
#     ax1.set_title(f"{feature}: Before vs After", fontsize=14, fontweight="bold")
#     ax1.set_xticks(x)
#     ax1.set_xticklabels(top_cats, rotation=45, ha="right", fontsize=9)
#     ax1.legend(fontsize=11)
#     ax1.grid(axis="y", alpha=0.3)

#     # 2. Change magnitude
#     ax2 = fig.add_subplot(gs[0, 2])
#     changes = dist_df.head(10)["change"] * 100
#     colors = ["#10b981" if x > 0 else "#ef4444" for x in changes]
#     ax2.barh(range(len(changes)), changes, color=colors, alpha=0.8)
#     ax2.set_yticks(range(len(changes)))
#     ax2.set_yticklabels(changes.index, fontsize=9)
#     ax2.axvline(0, color="black", linewidth=1.5)
#     ax2.set_xlabel("Change (%)", fontsize=11, fontweight="bold")
#     ax2.set_title("Absolute Change", fontsize=12, fontweight="bold")
#     ax2.grid(axis="x", alpha=0.3)

#     # 3. Drift type breakdown
#     ax3 = fig.add_subplot(gs[1, 0])
#     drift_counts = dist_df["drift_type"].value_counts()
#     colors_pie = {
#         "Increasing": "#10b981",
#         "Decreasing": "#ef4444",
#         "Stable": "#94a3b8",
#         "NEW": "#8b5cf6",
#         "DISAPPEARED": "#f59e0b",
#     }
#     pie_colors = [colors_pie.get(x, "#94a3b8") for x in drift_counts.index]
#     wedges, texts, autotexts = ax3.pie(
#         drift_counts.values,
#         labels=drift_counts.index,
#         autopct="%1.1f%%",
#         colors=pie_colors,
#         startangle=90,
#     )
#     for autotext in autotexts:
#         autotext.set_color("white")
#         autotext.set_fontweight("bold")
#     ax3.set_title("Drift Type Breakdown", fontsize=12, fontweight="bold")

#     # 4. Relative change (%)
#     ax4 = fig.add_subplot(gs[1, 1])
#     pct_changes = dist_df.head(10)["pct_change"].replace([np.inf, -np.inf], 0)
#     colors_pct = ["#10b981" if x > 0 else "#ef4444" for x in pct_changes]
#     ax4.barh(range(len(pct_changes)), pct_changes, color=colors_pct, alpha=0.8)
#     ax4.set_yticks(range(len(pct_changes)))
#     ax4.set_yticklabels(pct_changes.index, fontsize=9)
#     ax4.axvline(0, color="black", linewidth=1.5)
#     ax4.set_xlabel("Relative Change (%)", fontsize=11, fontweight="bold")
#     ax4.set_title("% Change from Baseline", fontsize=12, fontweight="bold")
#     ax4.grid(axis="x", alpha=0.3)

#     # 5. Cumulative distribution
#     ax5 = fig.add_subplot(gs[1, 2])
#     sorted1 = dist1.sort_values(ascending=False).cumsum() * 100
#     sorted2 = dist2.sort_values(ascending=False).cumsum() * 100

#     ax5.plot(
#         range(len(sorted1[:30])),
#         sorted1[:30].values,
#         label=month1,
#         linewidth=2.5,
#         marker="o",
#         markersize=5,
#         color="#3b82f6",
#     )
#     ax5.plot(
#         range(len(sorted2[:30])),
#         sorted2[:30].values,
#         label=month2,
#         linewidth=2.5,
#         marker="s",
#         markersize=5,
#         color="#ef4444",
#     )
#     ax5.axhline(80, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
#     ax5.set_xlabel("Category Rank", fontsize=11, fontweight="bold")
#     ax5.set_ylabel("Cumulative %", fontsize=11, fontweight="bold")
#     ax5.set_title("Cumulative Distribution", fontsize=12, fontweight="bold")
#     ax5.legend(fontsize=10)
#     ax5.grid(alpha=0.3)

#     plt.suptitle(
#         f"Drift Deep Dive: {feature}\nWhy it's changing from {month1} to {month2}",
#         fontsize=16,
#         fontweight="bold",
#     )

#     plt.savefig(f"drift_deepdive_{feature}.png", dpi=300, bbox_inches="tight")
#     print(f"\n📊 Saved: drift_deepdive_{feature}.png")
#     plt.show()

#     # Root cause summary
#     print("\n" + "=" * 80)
#     print("🔍 ROOT CAUSE SUMMARY")
#     print("=" * 80)

#     if len(new_cats) > 3:
#         print(
#             f"⚠️  Many new categories ({len(new_cats)}) → Possible data source change or expansion"
#         )
#     if len(disappeared) > 3:
#         print(
#             f"⚠️  Many disappeared categories ({len(disappeared)}) → Possible discontinuation or data quality issue"
#         )

#     # Check concentration
#     herfindahl1 = (dist1**2).sum()
#     herfindahl2 = (dist2**2).sum()
#     if herfindahl2 > herfindahl1 * 1.15:
#         print(f"⚠️  Distribution MORE concentrated → Few categories dominating")
#     elif herfindahl2 < herfindahl1 * 0.85:
#         print(f"⚠️  Distribution LESS concentrated → More diversity/fragmentation")

#     # Check if systematic shift
#     if len(increasing) > len(decreasing) * 1.5:
#         print(
#             f"⚠️  More categories growing than shrinking → Possible expansion/growth pattern"
#         )
#     elif len(decreasing) > len(increasing) * 1.5:
#         print(
#             f"⚠️  More categories shrinking than growing → Possible contraction/consolidation"
#         )

#     print("\n💡 RECOMMENDATION:")
#     if tvd > 0.15:
#         print("   🚨 CRITICAL: Immediate model retraining needed")
#         print("   🔎 Investigate business/operational changes in data collection")
#     elif tvd > 0.08:
#         print("   ⚠️  Schedule model retraining soon")
#         print("   📊 Monitor this feature closely in upcoming periods")
#     else:
#         print("   ✓ Continue monitoring, retraining not urgent")

#     return dist_df

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


class CompleteDriftAnalysis:
    """
    Complete month-over-month drift analysis for both categorical and numerical features.
    Analyzes everything that's changing between start and end month with comprehensive visualizations.
    """

    def __init__(self, df, month_col, start_month, end_month):
        """
        Initialize complete drift analyzer.

        Parameters:
        -----------
        df : pd.DataFrame
            Your dataset
        month_col : str
            Name of the month/date column
        start_month : str
            Starting month (baseline)
        end_month : str
            Ending month (comparison)
        """
        self.df = df.copy()
        self.month_col = month_col
        self.start_month = start_month
        self.end_month = end_month

        # Filter data for the two periods
        self.df_start = df[df[month_col] == start_month].copy()
        self.df_end = df[df[month_col] == end_month].copy()

        # Identify feature types
        self.all_features = [col for col in df.columns if col != month_col]
        self.categorical_features = []
        self.numerical_features = []

        for feat in self.all_features:
            if df[feat].dtype == "object" or df[feat].nunique() < 20:
                self.categorical_features.append(feat)
            else:
                self.numerical_features.append(feat)

        self.results = {}

        print("=" * 80)
        print(f"COMPLETE DRIFT ANALYSIS: {start_month} → {end_month}")
        print("=" * 80)
        print(f"Start period: {len(self.df_start):,} rows")
        print(f"End period: {len(self.df_end):,} rows")
        print(f"Categorical features: {len(self.categorical_features)}")
        print(f"Numerical features: {len(self.numerical_features)}")
        print("=" * 80)

    def analyze_categorical_feature(self, feature):
        """Deep analysis of a categorical feature."""
        start_dist = self.df_start[feature].value_counts(normalize=True)
        end_dist = self.df_end[feature].value_counts(normalize=True)

        # Combine distributions
        dist_df = pd.concat(
            [start_dist, end_dist], axis=1, keys=[self.start_month, self.end_month]
        ).fillna(0)
        dist_df["change"] = dist_df[self.end_month] - dist_df[self.start_month]
        dist_df["abs_change"] = dist_df["change"].abs()
        dist_df["pct_change"] = (
            (dist_df[self.end_month] - dist_df[self.start_month])
            / (dist_df[self.start_month] + 1e-10)
        ) * 100

        # Classify changes
        dist_df["drift_type"] = "Stable"
        dist_df.loc[dist_df["change"] > 0.02, "drift_type"] = "Increasing"
        dist_df.loc[dist_df["change"] < -0.02, "drift_type"] = "Decreasing"
        dist_df.loc[
            (dist_df[self.start_month] == 0) & (dist_df[self.end_month] > 0),
            "drift_type",
        ] = "NEW"
        dist_df.loc[
            (dist_df[self.start_month] > 0) & (dist_df[self.end_month] == 0),
            "drift_type",
        ] = "DISAPPEARED"

        # Calculate metrics
        tvd = 0.5 * dist_df["abs_change"].sum()

        # Chi-square test
        all_cats = set(start_dist.index) | set(end_dist.index)
        observed = []
        for cat in all_cats:
            observed.append(
                [
                    (start_dist.get(cat, 0) * len(self.df_start)),
                    (end_dist.get(cat, 0) * len(self.df_end)),
                ]
            )
        observed = np.array(observed)

        try:
            chi2, p_value, _, _ = stats.chi2_contingency(observed.T)
        except:
            chi2, p_value = 0, 1.0

        # Jensen-Shannon divergence
        all_categories = sorted(all_cats)
        p = np.array([start_dist.get(cat, 1e-10) for cat in all_categories])
        q = np.array([end_dist.get(cat, 1e-10) for cat in all_categories])
        p = p / p.sum()
        q = q / q.sum()
        jsd = jensenshannon(p, q) ** 2

        return {
            "feature": feature,
            "type": "categorical",
            "dist_df": dist_df.sort_values("abs_change", ascending=False),
            "tvd": tvd,
            "chi2": chi2,
            "p_value": p_value,
            "jsd": jsd,
            "drift_detected": tvd > 0.1 or p_value < 0.05,
        }

    def analyze_numerical_feature(self, feature):
        """Deep analysis of a numerical feature."""
        start_vals = self.df_start[feature].dropna()
        end_vals = self.df_end[feature].dropna()

        # Statistical tests
        ks_stat, ks_p = stats.ks_2samp(start_vals, end_vals)
        t_stat, t_p = stats.ttest_ind(start_vals, end_vals)

        # Descriptive statistics
        stats_start = {
            "mean": start_vals.mean(),
            "median": start_vals.median(),
            "std": start_vals.std(),
            "min": start_vals.min(),
            "max": start_vals.max(),
            "q25": start_vals.quantile(0.25),
            "q75": start_vals.quantile(0.75),
        }

        stats_end = {
            "mean": end_vals.mean(),
            "median": end_vals.median(),
            "std": end_vals.std(),
            "min": end_vals.min(),
            "max": end_vals.max(),
            "q25": end_vals.quantile(0.25),
            "q75": end_vals.quantile(0.75),
        }

        # Calculate changes
        changes = {
            "mean_change": stats_end["mean"] - stats_start["mean"],
            "mean_pct_change": (
                (stats_end["mean"] - stats_start["mean"])
                / (stats_start["mean"] + 1e-10)
            )
            * 100,
            "median_change": stats_end["median"] - stats_start["median"],
            "std_change": stats_end["std"] - stats_start["std"],
        }

        return {
            "feature": feature,
            "type": "numerical",
            "stats_start": stats_start,
            "stats_end": stats_end,
            "changes": changes,
            "ks_stat": ks_stat,
            "ks_p": ks_p,
            "t_stat": t_stat,
            "t_p": t_p,
            "drift_detected": ks_p < 0.05 or abs(changes["mean_pct_change"]) > 10,
        }

    def run_complete_analysis(self):
        """Run complete analysis on all features."""
        print("\n" + "=" * 80)
        print("ANALYZING ALL FEATURES")
        print("=" * 80)

        # Analyze categorical features
        print(
            f"\n📊 Analyzing {len(self.categorical_features)} categorical features..."
        )
        for feat in self.categorical_features:
            self.results[feat] = self.analyze_categorical_feature(feat)
            status = (
                "🚨 DRIFTING" if self.results[feat]["drift_detected"] else "✓ Stable"
            )
            print(f"   {feat}: {status}")

        # Analyze numerical features
        print(f"\n📈 Analyzing {len(self.numerical_features)} numerical features...")
        for feat in self.numerical_features:
            self.results[feat] = self.analyze_numerical_feature(feat)
            status = (
                "🚨 DRIFTING" if self.results[feat]["drift_detected"] else "✓ Stable"
            )
            print(f"   {feat}: {status}")

        self._print_summary()

    def _print_summary(self):
        """Print comprehensive summary."""
        print("\n" + "=" * 80)
        print("DRIFT SUMMARY")
        print("=" * 80)

        # Categorical summary
        cat_drifting = [
            f for f in self.categorical_features if self.results[f]["drift_detected"]
        ]
        cat_stable = [
            f
            for f in self.categorical_features
            if not self.results[f]["drift_detected"]
        ]

        print(f"\n📊 CATEGORICAL FEATURES:")
        print(f"   🚨 {len(cat_drifting)} features drifting")
        print(f"   ✓ {len(cat_stable)} features stable")

        if cat_drifting:
            print("\n   Drifting features (by severity):")
            cat_sorted = sorted(
                cat_drifting, key=lambda x: self.results[x]["tvd"], reverse=True
            )
            for feat in cat_sorted[:10]:
                tvd = self.results[feat]["tvd"]
                print(f"      • {feat}: TVD={tvd:.4f}")

        # Numerical summary
        num_drifting = [
            f for f in self.numerical_features if self.results[f]["drift_detected"]
        ]
        num_stable = [
            f for f in self.numerical_features if not self.results[f]["drift_detected"]
        ]

        print(f"\n📈 NUMERICAL FEATURES:")
        print(f"   🚨 {len(num_drifting)} features drifting")
        print(f"   ✓ {len(num_stable)} features stable")

        if num_drifting:
            print("\n   Drifting features (by mean % change):")
            num_sorted = sorted(
                num_drifting,
                key=lambda x: abs(self.results[x]["changes"]["mean_pct_change"]),
                reverse=True,
            )
            for feat in num_sorted[:10]:
                change = self.results[feat]["changes"]["mean_pct_change"]
                print(f"      • {feat}: {change:+.2f}% mean change")

        print("\n" + "=" * 80)
        print("OVERALL ASSESSMENT")
        print("=" * 80)

        total_drifting = len(cat_drifting) + len(num_drifting)
        total_features = len(self.all_features)
        drift_pct = (total_drifting / total_features) * 100

        print(
            f"\n{total_drifting}/{total_features} features drifting ({drift_pct:.1f}%)"
        )

        if drift_pct > 50:
            print(
                "\n🚨 CRITICAL: Majority of features drifting - major distribution shift!"
            )
            print("   → Immediate model retraining required")
            print("   → Investigate data collection changes")
        elif drift_pct > 25:
            print("\n⚠️  SIGNIFICANT: Notable drift detected")
            print("   → Schedule model retraining")
            print("   → Review feature engineering")
        else:
            print("\n✓ MODERATE: Some drift but manageable")
            print("   → Continue monitoring")

    def plot_categorical_details(self, feature, top_n=15):
        """Detailed visualization for a categorical feature."""
        if (
            feature not in self.results
            or self.results[feature]["type"] != "categorical"
        ):
            print(f"Feature {feature} not found or not categorical!")
            return

        result = self.results[feature]
        dist_df = result["dist_df"]

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

        # 1. Before/After comparison
        ax1 = fig.add_subplot(gs[0, :2])
        top_cats = dist_df.head(top_n).index
        x = np.arange(len(top_cats))
        width = 0.35

        ax1.bar(
            x - width / 2,
            dist_df.loc[top_cats, self.start_month] * 100,
            width,
            label=self.start_month,
            alpha=0.8,
            color="#3b82f6",
        )
        ax1.bar(
            x + width / 2,
            dist_df.loc[top_cats, self.end_month] * 100,
            width,
            label=self.end_month,
            alpha=0.8,
            color="#ef4444",
        )
        ax1.set_ylabel("Percentage (%)", fontsize=12, fontweight="bold")
        ax1.set_title(f"{feature}: Distribution Change", fontsize=14, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(top_cats, rotation=45, ha="right", fontsize=9)
        ax1.legend(fontsize=11)
        ax1.grid(axis="y", alpha=0.3)

        # Add TVD annotation
        ax1.text(
            0.02,
            0.98,
            f'TVD: {result["tvd"]:.4f}\np-value: {result["p_value"]:.4f}',
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # 2. Absolute change
        ax2 = fig.add_subplot(gs[0, 2])
        changes = dist_df.head(10)["change"] * 100
        colors = ["#10b981" if x > 0 else "#ef4444" for x in changes]
        ax2.barh(range(len(changes)), changes, color=colors, alpha=0.8)
        ax2.set_yticks(range(len(changes)))
        ax2.set_yticklabels(changes.index, fontsize=9)
        ax2.axvline(0, color="black", linewidth=1.5)
        ax2.set_xlabel("Change (%)", fontsize=11, fontweight="bold")
        ax2.set_title("Absolute Change", fontsize=12, fontweight="bold")
        ax2.grid(axis="x", alpha=0.3)

        # 3. Drift type pie
        ax3 = fig.add_subplot(gs[1, 0])
        drift_counts = dist_df["drift_type"].value_counts()
        colors_pie = {
            "Increasing": "#10b981",
            "Decreasing": "#ef4444",
            "Stable": "#94a3b8",
            "NEW": "#8b5cf6",
            "DISAPPEARED": "#f59e0b",
        }
        pie_colors = [colors_pie.get(x, "#94a3b8") for x in drift_counts.index]
        wedges, texts, autotexts = ax3.pie(
            drift_counts.values,
            labels=drift_counts.index,
            autopct="%1.1f%%",
            colors=pie_colors,
            startangle=90,
        )
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
        ax3.set_title("Drift Type Distribution", fontsize=12, fontweight="bold")

        # 4. Relative change
        ax4 = fig.add_subplot(gs[1, 1])
        pct_changes = dist_df.head(10)["pct_change"].replace([np.inf, -np.inf], 0)
        colors_pct = ["#10b981" if x > 0 else "#ef4444" for x in pct_changes]
        ax4.barh(range(len(pct_changes)), pct_changes, color=colors_pct, alpha=0.8)
        ax4.set_yticks(range(len(pct_changes)))
        ax4.set_yticklabels(pct_changes.index, fontsize=9)
        ax4.axvline(0, color="black", linewidth=1.5)
        ax4.set_xlabel("% Change", fontsize=11, fontweight="bold")
        ax4.set_title("Relative Change", fontsize=12, fontweight="bold")
        ax4.grid(axis="x", alpha=0.3)

        # 5. Cumulative distribution
        ax5 = fig.add_subplot(gs[1, 2])
        start_dist = (
            self.df_start[feature]
            .value_counts(normalize=True)
            .sort_values(ascending=False)
        )
        end_dist = (
            self.df_end[feature]
            .value_counts(normalize=True)
            .sort_values(ascending=False)
        )
        cumsum_start = start_dist.cumsum() * 100
        cumsum_end = end_dist.cumsum() * 100

        ax5.plot(
            range(len(cumsum_start[:30])),
            cumsum_start[:30].values,
            label=self.start_month,
            linewidth=2.5,
            marker="o",
            markersize=5,
            color="#3b82f6",
        )
        ax5.plot(
            range(len(cumsum_end[:30])),
            cumsum_end[:30].values,
            label=self.end_month,
            linewidth=2.5,
            marker="s",
            markersize=5,
            color="#ef4444",
        )
        ax5.axhline(80, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
        ax5.set_xlabel("Category Rank", fontsize=11, fontweight="bold")
        ax5.set_ylabel("Cumulative %", fontsize=11, fontweight="bold")
        ax5.set_title("Cumulative Distribution", fontsize=12, fontweight="bold")
        ax5.legend(fontsize=10)
        ax5.grid(alpha=0.3)

        plt.suptitle(
            f"Categorical Feature Deep Dive: {feature}", fontsize=16, fontweight="bold"
        )
        plt.savefig(f"categorical_drift_{feature}.png", dpi=300, bbox_inches="tight")
        print(f"✓ Saved: categorical_drift_{feature}.png")
        plt.show()

    def plot_numerical_details(self, feature):
        """Detailed visualization for a numerical feature."""
        if feature not in self.results or self.results[feature]["type"] != "numerical":
            print(f"Feature {feature} not found or not numerical!")
            return

        result = self.results[feature]
        start_vals = self.df_start[feature].dropna()
        end_vals = self.df_end[feature].dropna()

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

        # 1. Distribution comparison (histogram)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.hist(
            start_vals,
            bins=50,
            alpha=0.6,
            label=self.start_month,
            color="#3b82f6",
            density=True,
        )
        ax1.hist(
            end_vals,
            bins=50,
            alpha=0.6,
            label=self.end_month,
            color="#ef4444",
            density=True,
        )
        ax1.set_xlabel(feature, fontsize=12, fontweight="bold")
        ax1.set_ylabel("Density", fontsize=12, fontweight="bold")
        ax1.set_title(
            f"{feature}: Distribution Comparison", fontsize=14, fontweight="bold"
        )
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3)

        # Add KS test result
        ax1.text(
            0.02,
            0.98,
            f'KS-test p-value: {result["ks_p"]:.4f}\nDrift: {"Yes" if result["drift_detected"] else "No"}',
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # 2. Box plot comparison
        ax2 = fig.add_subplot(gs[0, 2])
        box_data = [start_vals, end_vals]
        bp = ax2.boxplot(
            box_data,
            labels=[self.start_month, self.end_month],
            patch_artist=True,
            notch=True,
        )
        bp["boxes"][0].set_facecolor("#3b82f6")
        bp["boxes"][1].set_facecolor("#ef4444")
        ax2.set_ylabel(feature, fontsize=11, fontweight="bold")
        ax2.set_title("Box Plot Comparison", fontsize=12, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)

        # 3. QQ plot
        ax3 = fig.add_subplot(gs[1, 0])
        from scipy import stats as sp_stats

        sp_stats.probplot(start_vals, dist="norm", plot=ax3)
        ax3.set_title(f"{self.start_month} Q-Q Plot", fontsize=11, fontweight="bold")
        ax3.grid(alpha=0.3)

        # 4. Statistics comparison
        ax4 = fig.add_subplot(gs[1, 1])
        stat_names = ["Mean", "Median", "Std", "Min", "Max"]
        start_stats = [
            result["stats_start"]["mean"],
            result["stats_start"]["median"],
            result["stats_start"]["std"],
            result["stats_start"]["min"],
            result["stats_start"]["max"],
        ]
        end_stats = [
            result["stats_end"]["mean"],
            result["stats_end"]["median"],
            result["stats_end"]["std"],
            result["stats_end"]["min"],
            result["stats_end"]["max"],
        ]

        x = np.arange(len(stat_names))
        width = 0.35
        ax4.bar(
            x - width / 2,
            start_stats,
            width,
            label=self.start_month,
            alpha=0.8,
            color="#3b82f6",
        )
        ax4.bar(
            x + width / 2,
            end_stats,
            width,
            label=self.end_month,
            alpha=0.8,
            color="#ef4444",
        )
        ax4.set_xticks(x)
        ax4.set_xticklabels(stat_names, rotation=45)
        ax4.set_ylabel("Value", fontsize=11, fontweight="bold")
        ax4.set_title("Statistical Comparison", fontsize=12, fontweight="bold")
        ax4.legend()
        ax4.grid(axis="y", alpha=0.3)

        # 5. Change metrics
        ax5 = fig.add_subplot(gs[1, 2])
        change_metrics = [
            "Mean\nChange",
            "Mean %\nChange",
            "Median\nChange",
            "Std\nChange",
        ]
        change_values = [
            result["changes"]["mean_change"],
            result["changes"]["mean_pct_change"],
            result["changes"]["median_change"],
            result["changes"]["std_change"],
        ]

        colors_change = ["#10b981" if x > 0 else "#ef4444" for x in change_values]
        ax5.barh(change_metrics, change_values, color=colors_change, alpha=0.8)
        ax5.axvline(0, color="black", linewidth=1.5)
        ax5.set_xlabel("Change", fontsize=11, fontweight="bold")
        ax5.set_title("Change Metrics", fontsize=12, fontweight="bold")
        ax5.grid(axis="x", alpha=0.3)

        plt.suptitle(
            f"Numerical Feature Deep Dive: {feature}", fontsize=16, fontweight="bold"
        )
        plt.savefig(f"numerical_drift_{feature}.png", dpi=300, bbox_inches="tight")
        print(f"✓ Saved: numerical_drift_{feature}.png")
        plt.show()

    def plot_overall_summary(self):
        """Create overall summary dashboard."""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        # 1. Drift detection summary
        ax1 = fig.add_subplot(gs[0, :])

        cat_drifting = sum(
            [1 for f in self.categorical_features if self.results[f]["drift_detected"]]
        )
        cat_stable = len(self.categorical_features) - cat_drifting
        num_drifting = sum(
            [1 for f in self.numerical_features if self.results[f]["drift_detected"]]
        )
        num_stable = len(self.numerical_features) - num_drifting

        categories = [
            "Categorical\nDrifting",
            "Categorical\nStable",
            "Numerical\nDrifting",
            "Numerical\nStable",
        ]
        values = [cat_drifting, cat_stable, num_drifting, num_stable]
        colors = ["#ef4444", "#10b981", "#ef4444", "#10b981"]

        ax1.bar(categories, values, color=colors, alpha=0.8)
        ax1.set_ylabel("Count", fontsize=12, fontweight="bold")
        ax1.set_title("Overall Drift Detection Summary", fontsize=14, fontweight="bold")
        ax1.grid(axis="y", alpha=0.3)

        # Add value labels
        for i, v in enumerate(values):
            ax1.text(i, v + 0.5, str(v), ha="center", fontweight="bold", fontsize=12)

        # 2. Top drifting categorical features
        ax2 = fig.add_subplot(gs[1, :2])
        cat_tvds = [(f, self.results[f]["tvd"]) for f in self.categorical_features]
        cat_tvds_sorted = sorted(cat_tvds, key=lambda x: x[1], reverse=True)[:10]

        if cat_tvds_sorted:
            feats, tvds = zip(*cat_tvds_sorted)
            colors_tvd = [
                "#ef4444" if t > 0.2 else "#f59e0b" if t > 0.1 else "#10b981"
                for t in tvds
            ]
            ax2.barh(range(len(feats)), tvds, color=colors_tvd, alpha=0.8)
            ax2.set_yticks(range(len(feats)))
            ax2.set_yticklabels(feats, fontsize=9)
            ax2.axvline(
                0.1, color="orange", linestyle="--", alpha=0.7, label="Moderate"
            )
            ax2.axvline(0.2, color="red", linestyle="--", alpha=0.7, label="High")
            ax2.set_xlabel("Total Variation Distance", fontsize=11, fontweight="bold")
            ax2.set_title(
                "Top Drifting Categorical Features", fontsize=12, fontweight="bold"
            )
            ax2.legend()
            ax2.grid(axis="x", alpha=0.3)

        # 3. Drift severity pie
        ax3 = fig.add_subplot(gs[1, 2])
        severe = sum(
            [1 for f in self.categorical_features if self.results[f]["tvd"] > 0.2]
        )
        moderate = sum(
            [
                1
                for f in self.categorical_features
                if 0.1 < self.results[f]["tvd"] <= 0.2
            ]
        )
        mild = len(self.categorical_features) - severe - moderate

        pie_data = [severe, moderate, mild]
        pie_labels = [
            f"Severe\n({severe})",
            f"Moderate\n({moderate})",
            f"Mild\n({mild})",
        ]
        pie_colors = ["#ef4444", "#f59e0b", "#10b981"]

        ax3.pie(
            pie_data,
            labels=pie_labels,
            autopct="%1.1f%%",
            colors=pie_colors,
            startangle=90,
        )
        ax3.set_title("Categorical Drift Severity", fontsize=12, fontweight="bold")

        # 4. Top drifting numerical features
        ax4 = fig.add_subplot(gs[2, :2])
        num_changes = [
            (f, abs(self.results[f]["changes"]["mean_pct_change"]))
            for f in self.numerical_features
        ]
        num_changes_sorted = sorted(num_changes, key=lambda x: x[1], reverse=True)[:10]

        if num_changes_sorted:
            feats_num, changes = zip(*num_changes_sorted)
            colors_num = [
                "#ef4444" if c > 20 else "#f59e0b" if c > 10 else "#10b981"
                for c in changes
            ]
            ax4.barh(range(len(feats_num)), changes, color=colors_num, alpha=0.8)
            ax4.set_yticks(range(len(feats_num)))
            ax4.set_yticklabels(feats_num, fontsize=9)
            ax4.axvline(10, color="orange", linestyle="--", alpha=0.7, label="Moderate")
            ax4.axvline(20, color="red", linestyle="--", alpha=0.7, label="High")
            ax4.set_xlabel("|Mean % Change|", fontsize=11, fontweight="bold")
            ax4.set_title(
                "Top Drifting Numerical Features", fontsize=12, fontweight="bold"
            )
            ax4.legend()
            ax4.grid(axis="x", alpha=0.3)

        # 5. Summary text
        ax5 = fig.add_subplot(gs[2, 2])
        ax5.axis("off")

        total_features = len(self.all_features)
        total_drifting = cat_drifting + num_drifting
        drift_pct = (total_drifting / total_features) * 100

        summary_text = f"""
OVERALL SUMMARY
{'='*30}

Total Features: {total_features}
Drifting: {total_drifting} ({drift_pct:.1f}%)
Stable: {total_features - total_drifting} ({100-drift_pct:.1f}%)

CATEGORICAL:
  Drifting: {cat_drifting}/{len(self.categorical_features)}
  Stable: {cat_stable}/{len(self.categorical_features)}

NUMERICAL:
  Drifting: {num_drifting}/{len(self.numerical_features)}
  Stable: {num_stable}/{len(self.numerical_features)}

RECOMMENDATION:
"""

        if drift_pct > 50:
            summary_text += "  🚨 CRITICAL - Immediate\n  model retraining needed!"
        elif drift_pct > 25:
            summary_text += "  ⚠️ SIGNIFICANT - Schedule\n  retraining soon"
        else:
            summary_text += "  ✓ MODERATE - Continue\n  monitoring"

        ax5.text(
            0.1,
            0.9,
            summary_text,
            fontsize=11,
            verticalalignment="top",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.suptitle(
            f"Complete Drift Analysis Dashboard\n{self.start_month} → {self.end_month}",
            fontsize=18,
            fontweight="bold",
        )

        plt.savefig("complete_drift_dashboard.png", dpi=300, bbox_inches="tight")
        print("\n✓ Saved: complete_drift_dashboard.png")
        plt.show()

    def generate_detailed_report(self):
        """Generate detailed text report of all changes."""
        print("\n" + "=" * 80)
        print("DETAILED DRIFT REPORT")
        print(f"{self.start_month} → {self.end_month}")
        print("=" * 80)

        # Categorical features report
        print("\n" + "=" * 80)
        print("CATEGORICAL FEATURES ANALYSIS")
        print("=" * 80)

        for feat in self.categorical_features:
            result = self.results[feat]
            if result["drift_detected"]:
                print(f"\n🚨 {feat} - DRIFTING")
                print("-" * 80)
                print(f"   Total Variation Distance: {result['tvd']:.4f}")
                print(f"   Chi-square p-value: {result['p_value']:.6f}")
                print(f"   JS Divergence: {result['jsd']:.4f}")

                dist_df = result["dist_df"]

                # Show top changes
                print("\n   TOP 5 CHANGES:")
                for i, (cat, row) in enumerate(dist_df.head(5).iterrows(), 1):
                    start_pct = row[self.start_month] * 100
                    end_pct = row[self.end_month] * 100
                    change = row["change"] * 100
                    dtype = row["drift_type"]

                    arrow = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                    print(f"   {i}. {cat} {arrow} {dtype}")
                    print(
                        f"      {self.start_month}: {start_pct:.2f}% → {self.end_month}: {end_pct:.2f}% ({change:+.2f}%)"
                    )

                # Count by drift type
                new_cats = dist_df[dist_df["drift_type"] == "NEW"]
                disappeared = dist_df[dist_df["drift_type"] == "DISAPPEARED"]

                if len(new_cats) > 0:
                    print(f"\n   ✨ {len(new_cats)} NEW categories appeared")
                if len(disappeared) > 0:
                    print(f"   ❌ {len(disappeared)} categories DISAPPEARED")

        # Numerical features report
        print("\n" + "=" * 80)
        print("NUMERICAL FEATURES ANALYSIS")
        print("=" * 80)

        for feat in self.numerical_features:
            result = self.results[feat]
            if result["drift_detected"]:
                print(f"\n🚨 {feat} - DRIFTING")
                print("-" * 80)
                print(f"   KS-test p-value: {result['ks_p']:.6f}")
                print(f"   T-test p-value: {result['t_p']:.6f}")

                stats_start = result["stats_start"]
                stats_end = result["stats_end"]
                changes = result["changes"]

                print(f"\n   STATISTICS COMPARISON:")
                print(
                    f"   {'Metric':<12} {self.start_month:>12} {self.end_month:>12} {'Change':>12} {'% Change':>12}"
                )
                print(f"   {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

                metrics_to_show = ["mean", "median", "std", "min", "max"]
                for metric in metrics_to_show:
                    start_val = stats_start[metric]
                    end_val = stats_end[metric]
                    change_val = end_val - start_val
                    pct_change = (change_val / (start_val + 1e-10)) * 100

                    print(
                        f"   {metric:<12} {start_val:>12.2f} {end_val:>12.2f} {change_val:>+12.2f} {pct_change:>+11.2f}%"
                    )

                print(f"\n   KEY INSIGHTS:")
                if abs(changes["mean_pct_change"]) > 20:
                    print(
                        f"      🚨 Mean shifted by {changes['mean_pct_change']:+.1f}% - MAJOR CHANGE"
                    )
                elif abs(changes["mean_pct_change"]) > 10:
                    print(
                        f"      ⚠️ Mean shifted by {changes['mean_pct_change']:+.1f}% - NOTABLE CHANGE"
                    )

                if abs(changes["std_change"]) / (stats_start["std"] + 1e-10) > 0.2:
                    print(f"      ⚠️ Variability changed significantly")

        print("\n" + "=" * 80)
        print("REPORT COMPLETE")
        print("=" * 80)


# Example usage and demonstration
if __name__ == "__main__":
    # Create sample dataset with drift
    np.random.seed(42)

    # Generate data for January (baseline)
    jan_data = []
    for _ in range(2000):
        jan_data.append(
            {
                "month": "Jan",
                "category_A": np.random.choice(["X", "Y", "Z"], p=[0.5, 0.3, 0.2]),
                "category_B": np.random.choice(
                    ["A", "B", "C", "D"], p=[0.4, 0.3, 0.2, 0.1]
                ),
                "numeric_1": np.random.normal(100, 15),
                "numeric_2": np.random.exponential(50),
                "numeric_3": np.random.uniform(0, 100),
            }
        )

    # Generate data for June (with drift)
    jun_data = []
    for _ in range(2000):
        jun_data.append(
            {
                "month": "Jun",
                "category_A": np.random.choice(
                    ["X", "Y", "Z", "W"], p=[0.3, 0.3, 0.25, 0.15]
                ),  # New category 'W'
                "category_B": np.random.choice(
                    ["A", "B", "C", "E"], p=[0.2, 0.4, 0.3, 0.1]
                ),  # 'D' disappeared, 'E' new
                "numeric_1": np.random.normal(120, 20),  # Mean and std changed
                "numeric_2": np.random.exponential(65),  # Mean increased
                "numeric_3": np.random.uniform(10, 90),  # Range shifted
            }
        )

    # Combine data
    df = pd.DataFrame(jan_data + jun_data)

    print("Sample Dataset Created")
    print("=" * 80)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Months: {df['month'].unique()}")

    # Initialize complete drift analyzer
    analyzer = CompleteDriftAnalysis(
        df=df, month_col="month", start_month="Jan", end_month="Jun"
    )

    # Run complete analysis
    analyzer.run_complete_analysis()

    # Generate overall dashboard
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    analyzer.plot_overall_summary()

    # Generate detailed plots for drifting features
    print("\nGenerating detailed plots for drifting features...")

    # Plot categorical features
    for feat in analyzer.categorical_features:
        if analyzer.results[feat]["drift_detected"]:
            analyzer.plot_categorical_details(feat)

    # Plot numerical features
    for feat in analyzer.numerical_features:
        if analyzer.results[feat]["drift_detected"]:
            analyzer.plot_numerical_details(feat)

    # Generate detailed text report
    analyzer.generate_detailed_report()

    print("\n" + "=" * 80)
    print("✓ COMPLETE ANALYSIS FINISHED!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  • complete_drift_dashboard.png - Overall summary")
    print("  • categorical_drift_*.png - Categorical feature details")
    print("  • numerical_drift_*.png - Numerical feature details")
    print("\nUse these to understand exactly what's changing in your data!")
