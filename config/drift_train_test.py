import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")


class TrainTestDriftAnalysis:
    """
    Comprehensive train vs test drift analysis with PDF report generation.
    Identifies distribution differences, new categories, and data quality issues.
    """

    def __init__(self, train_df, test_df, dataset_names=None):
        """
        Initialize train-test drift analyzer.

        Parameters:
        -----------
        train_df : pd.DataFrame
            Training dataset
        test_df : pd.DataFrame
            Test dataset
        dataset_names : tuple, optional
            Names for the datasets (default: ('Train', 'Test'))
        """
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
        self.dataset_names = dataset_names or ("Train", "Test")

        print(f"\n{'='*80}")
        print(f"TRAIN vs TEST DRIFT ANALYSIS")
        print(f"{'='*80}")
        print(f"{self.dataset_names[0]} size: {len(train_df):,} rows")
        print(f"{self.dataset_names[1]} size: {len(test_df):,} rows")

        # Get common columns
        self.common_features = list(set(train_df.columns) & set(test_df.columns))
        print(f"Common features: {len(self.common_features)}")

        # Identify feature types
        self.categorical_features = []
        self.numerical_features = []

        for feat in self.common_features:
            if train_df[feat].dtype == "object" or train_df[feat].nunique() < 20:
                self.categorical_features.append(feat)
            else:
                self.numerical_features.append(feat)

        print(f"Categorical features: {len(self.categorical_features)}")
        print(f"Numerical features: {len(self.numerical_features)}")
        print(f"{'='*80}\n")

        self.results = {}

    def analyze_categorical_feature(self, feature):
        """Analyze a categorical feature for train-test drift."""
        train_vals = self.train_df[feature].dropna()
        test_vals = self.test_df[feature].dropna()

        train_dist = train_vals.value_counts(normalize=True)
        test_dist = test_vals.value_counts(normalize=True)

        train_cats = set(train_vals.unique())
        test_cats = set(test_vals.unique())

        # Find new and missing categories
        new_categories = test_cats - train_cats
        missing_categories = train_cats - test_cats
        common_categories = train_cats & test_cats

        # Calculate TVD
        all_cats = train_cats | test_cats
        tvd = 0.5 * sum(
            [abs(train_dist.get(cat, 0) - test_dist.get(cat, 0)) for cat in all_cats]
        )

        # Calculate JSD
        all_categories = sorted(all_cats)
        p = np.array([train_dist.get(cat, 1e-10) for cat in all_categories])
        q = np.array([test_dist.get(cat, 1e-10) for cat in all_categories])
        p = p / p.sum()
        q = q / q.sum()
        jsd = jensenshannon(p, q) ** 2

        # Chi-square test
        observed = []
        for cat in all_cats:
            observed.append(
                [
                    (train_dist.get(cat, 0) * len(train_vals)),
                    (test_dist.get(cat, 0) * len(test_vals)),
                ]
            )
        observed = np.array(observed)

        try:
            chi2, p_value, _, _ = stats.chi2_contingency(observed.T)
        except:
            chi2, p_value = 0, 1.0

        # Calculate impact of new categories
        new_cat_proportion = 0
        if len(new_categories) > 0:
            new_cat_proportion = sum([test_dist.get(cat, 0) for cat in new_categories])

        return {
            "feature": feature,
            "type": "categorical",
            "train_unique": len(train_cats),
            "test_unique": len(test_cats),
            "common_unique": len(common_categories),
            "new_categories": list(new_categories),
            "new_categories_count": len(new_categories),
            "missing_categories": list(missing_categories),
            "missing_categories_count": len(missing_categories),
            "new_cat_proportion": new_cat_proportion,
            "tvd": tvd,
            "jsd": jsd,
            "chi2": chi2,
            "p_value": p_value,
            "drift_detected": tvd > 0.1 or p_value < 0.05 or len(new_categories) > 0,
            "train_dist": train_dist,
            "test_dist": test_dist,
            "train_vals": train_vals,
            "test_vals": test_vals,
        }

    def analyze_numerical_feature(self, feature):
        """Analyze a numerical feature for train-test drift."""
        train_vals = self.train_df[feature].dropna()
        test_vals = self.test_df[feature].dropna()

        # Statistical tests
        ks_stat, ks_p = stats.ks_2samp(train_vals, test_vals)
        t_stat, t_p = stats.ttest_ind(train_vals, test_vals)

        # Statistics
        train_mean, test_mean = train_vals.mean(), test_vals.mean()
        train_median, test_median = train_vals.median(), test_vals.median()
        train_std, test_std = train_vals.std(), test_vals.std()
        train_min, test_min = train_vals.min(), test_vals.min()
        train_max, test_max = train_vals.max(), test_vals.max()

        mean_change = test_mean - train_mean
        mean_pct_change = (mean_change / (train_mean + 1e-10)) * 100

        # Range comparison
        train_range = train_max - train_min
        test_range = test_max - test_min

        # Out of bounds detection
        test_below_train_min = (test_vals < train_min).sum()
        test_above_train_max = (test_vals > train_max).sum()
        out_of_bounds_pct = 0
        if len(test_vals) > 0:
            out_of_bounds_pct = (
                (test_below_train_min + test_above_train_max) / len(test_vals) * 100
            )

        return {
            "feature": feature,
            "type": "numerical",
            "train_mean": train_mean,
            "test_mean": test_mean,
            "train_median": train_median,
            "test_median": test_median,
            "train_std": train_std,
            "test_std": test_std,
            "train_min": train_min,
            "test_min": test_min,
            "train_max": train_max,
            "test_max": test_max,
            "mean_change": mean_change,
            "mean_pct_change": mean_pct_change,
            "ks_stat": ks_stat,
            "ks_p": ks_p,
            "t_stat": t_stat,
            "t_p": t_p,
            "test_below_train_min": test_below_train_min,
            "test_above_train_max": test_above_train_max,
            "out_of_bounds_pct": out_of_bounds_pct,
            "drift_detected": ks_p < 0.05
            or abs(mean_pct_change) > 10
            or out_of_bounds_pct > 5,
            "train_vals": train_vals,
            "test_vals": test_vals,
        }

    def run_complete_analysis(self):
        """Run analysis on all features."""
        print("Analyzing all features for train-test drift...")
        print(f"{'='*80}\n")

        # Analyze categorical features
        print(f"📊 Analyzing {len(self.categorical_features)} categorical features...")
        for feat in self.categorical_features:
            result = self.analyze_categorical_feature(feat)
            self.results[feat] = result

            status = "✓ Stable"
            if result["new_categories_count"] > 0:
                status = f"🚨 {result['new_categories_count']} NEW categories ({result['new_cat_proportion']*100:.1f}% of test)"
            elif result["drift_detected"]:
                status = f"⚠️ Drift detected (TVD: {result['tvd']:.3f})"

            print(f"    {feat}: {status}")

        # Analyze numerical features
        print(f"\n📈 Analyzing {len(self.numerical_features)} numerical features...")
        for feat in self.numerical_features:
            result = self.analyze_numerical_feature(feat)
            self.results[feat] = result

            status = "✓ Stable"
            if result["out_of_bounds_pct"] > 5:
                status = f"🚨 {result['out_of_bounds_pct']:.1f}% out of train range"
            elif result["drift_detected"]:
                status = f"⚠️ Drift detected (KS p: {result['ks_p']:.3f})"

            print(f"    {feat}: {status}")

        print(f"\n{'='*80}")
        print("✓ Analysis complete!")
        print(f"{'='*80}\n")

    def create_categorical_viz(self, feature):
        """Create visualization for a categorical feature."""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

        result = self.results[feature]

        # 1. Distribution Comparison
        ax1 = fig.add_subplot(gs[0, :])

        # Get top categories by frequency in train
        top_cats = result["train_dist"].nlargest(15).index.tolist()
        # Add any new categories
        for cat in result["new_categories"][:5]:  # Top 5 new categories
            if cat not in top_cats:
                top_cats.append(cat)

        train_values = [result["train_dist"].get(cat, 0) * 100 for cat in top_cats]
        test_values = [result["test_dist"].get(cat, 0) * 100 for cat in top_cats]

        x = np.arange(len(top_cats))
        width = 0.35

        bars1 = ax1.bar(
            x - width / 2,
            train_values,
            width,
            label=self.dataset_names[0],
            alpha=0.8,
            color="#3b82f6",
        )
        bars2 = ax1.bar(
            x + width / 2,
            test_values,
            width,
            label=self.dataset_names[1],
            alpha=0.8,
            color="#10b981",
        )

        # Highlight new categories
        for i, cat in enumerate(top_cats):
            if cat in result["new_categories"]:
                bars2[i].set_color("#ef4444")
                bars2[i].set_label(
                    "New in Test"
                    if i == top_cats.index(result["new_categories"][0])
                    else ""
                )

        ax1.set_xlabel("Category", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Percentage (%)", fontsize=13, fontweight="bold")
        ax1.set_title(
            f"{feature}: Distribution Comparison", fontsize=15, fontweight="bold"
        )
        ax1.set_xticks(x)
        ax1.set_xticklabels(top_cats, rotation=45, ha="right")
        ax1.legend(fontsize=12)
        ax1.grid(axis="y", alpha=0.3)

        # 2. Category Changes
        ax2 = fig.add_subplot(gs[1, 0])

        categories = ["Common", "New in Test", "Missing in Test"]
        counts = [
            result["common_unique"],
            result["new_categories_count"],
            result["missing_categories_count"],
        ]
        colors = ["#10b981", "#ef4444", "#f59e0b"]

        wedges, texts, autotexts = ax2.pie(
            counts, labels=categories, autopct="%1.1f%%", colors=colors, startangle=90
        )
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontsize(12)
            autotext.set_fontweight("bold")

        ax2.set_title("Category Composition", fontsize=13, fontweight="bold")

        # 3. New Categories Detail
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis("off")

        new_cat_text = f"NEW CATEGORIES IN TEST\n{'='*40}\n\n"
        if result["new_categories_count"] > 0:
            new_cat_text += f"Total new categories: {result['new_categories_count']}\n"
            new_cat_text += (
                f"Proportion of test data: {result['new_cat_proportion']*100:.2f}%\n\n"
            )
            new_cat_text += "Top new categories:\n"
            for i, cat in enumerate(result["new_categories"][:10], 1):
                freq = result["test_dist"].get(cat, 0) * 100
                new_cat_text += f"  {i}. {cat}: {freq:.2f}%\n"
            if result["new_categories_count"] > 10:
                new_cat_text += f"  ... and {result['new_categories_count']-10} more\n"
        else:
            new_cat_text += "✓ No new categories in test set"

        ax3.text(
            0.1,
            0.9,
            new_cat_text,
            fontsize=11,
            verticalalignment="top",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # 4. Drift Metrics
        ax4 = fig.add_subplot(gs[2, 0])

        metrics = ["TVD", "JSD", "Chi² p-value"]
        values = [result["tvd"], result["jsd"], result["p_value"]]
        colors_bar = [
            "#ef4444" if result["tvd"] > 0.1 else "#10b981",
            "#ef4444" if result["jsd"] > 0.1 else "#10b981",
            "#ef4444" if result["p_value"] < 0.05 else "#10b981",
        ]

        ax4.barh(metrics, values, color=colors_bar, alpha=0.8)
        ax4.set_xlabel("Value", fontsize=12, fontweight="bold")
        ax4.set_title("Drift Metrics", fontsize=13, fontweight="bold")
        ax4.grid(axis="x", alpha=0.3)

        # 5. Summary
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis("off")

        summary_text = f"""
SUMMARY STATISTICS
{'='*40}

Feature: {feature}
Type: Categorical

CATEGORY COUNTS:
  • {self.dataset_names[0]}: {result['train_unique']} unique
  • {self.dataset_names[1]}: {result['test_unique']} unique
  • Common: {result['common_unique']}
  • New in test: {result['new_categories_count']}
  • Missing in test: {result['missing_categories_count']}

DRIFT METRICS:
  • TVD: {result['tvd']:.4f}
  • JSD: {result['jsd']:.4f}
  • Chi² p-value: {result['p_value']:.4f}

ASSESSMENT:
"""

        if result["new_categories_count"] > 0 and result["new_cat_proportion"] > 0.1:
            summary_text += f"  🚨 CRITICAL - {result['new_categories_count']} new\n"
            summary_text += (
                f"  categories ({result['new_cat_proportion']*100:.1f}% of test)"
            )
        elif result["new_categories_count"] > 0:
            summary_text += f"  ⚠️ WARNING - {result['new_categories_count']} new\n"
            summary_text += (
                f"  categories ({result['new_cat_proportion']*100:.1f}% of test)"
            )
        elif result["tvd"] > 0.2:
            summary_text += "  🚨 CRITICAL - High distribution\n  shift"
        elif result["tvd"] > 0.1:
            summary_text += "  ⚠️ MODERATE - Distribution\n  differences detected"
        else:
            summary_text += "  ✓ STABLE - Minor variations"

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
            f"Categorical Feature Analysis: {feature}\nTrain vs Test Comparison",
            fontsize=17,
            fontweight="bold",
        )

        return fig

    def create_numerical_viz(self, feature):
        """Create visualization for a numerical feature."""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

        result = self.results[feature]

        # 1. Distribution Comparison (Histogram)
        ax1 = fig.add_subplot(gs[0, :])

        ax1.hist(
            result["train_vals"],
            bins=50,
            alpha=0.6,
            label=self.dataset_names[0],
            color="#3b82f6",
            density=True,
        )
        ax1.hist(
            result["test_vals"],
            bins=50,
            alpha=0.6,
            label=self.dataset_names[1],
            color="#10b981",
            density=True,
        )
        ax1.axvline(
            result["train_mean"],
            color="#3b82f6",
            linestyle="--",
            linewidth=2,
            label=f"{self.dataset_names[0]} mean",
        )
        ax1.axvline(
            result["test_mean"],
            color="#10b981",
            linestyle="--",
            linewidth=2,
            label=f"{self.dataset_names[1]} mean",
        )
        ax1.set_xlabel("Value", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Density", fontsize=13, fontweight="bold")
        ax1.set_title(
            f"{feature}: Distribution Comparison", fontsize=15, fontweight="bold"
        )
        ax1.legend(fontsize=12)
        ax1.grid(alpha=0.3)

        # 2. Box Plots
        ax2 = fig.add_subplot(gs[1, 0])

        bp = ax2.boxplot(
            [result["train_vals"], result["test_vals"]],
            labels=self.dataset_names,
            patch_artist=True,
        )
        bp["boxes"][0].set_facecolor("#3b82f6")
        bp["boxes"][1].set_facecolor("#10b981")

        ax2.set_ylabel("Value", fontsize=12, fontweight="bold")
        ax2.set_title("Distribution Shape Comparison", fontsize=13, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)

        # 3. Statistics Comparison
        ax3 = fig.add_subplot(gs[1, 1])

        stats_labels = ["Mean", "Median", "Std Dev", "Min", "Max"]
        train_stats = [
            result["train_mean"],
            result["train_median"],
            result["train_std"],
            result["train_min"],
            result["train_max"],
        ]
        test_stats = [
            result["test_mean"],
            result["test_median"],
            result["test_std"],
            result["test_min"],
            result["test_max"],
        ]

        x = np.arange(len(stats_labels))
        width = 0.35

        ax3.bar(
            x - width / 2,
            train_stats,
            width,
            label=self.dataset_names[0],
            alpha=0.8,
            color="#3b82f6",
        )
        ax3.bar(
            x + width / 2,
            test_stats,
            width,
            label=self.dataset_names[1],
            alpha=0.8,
            color="#10b981",
        )

        ax3.set_ylabel("Value", fontsize=12, fontweight="bold")
        ax3.set_title("Statistical Summary", fontsize=13, fontweight="bold")
        ax3.set_xticks(x)
        ax3.set_xticklabels(stats_labels, rotation=45, ha="right")
        ax3.legend(fontsize=12)
        ax3.grid(axis="y", alpha=0.3)

        # 4. Out of Bounds Analysis
        ax4 = fig.add_subplot(gs[2, 0])

        categories = ["Within Range", "Below Min", "Above Max"]
        within = (
            len(result["test_vals"])
            - result["test_below_train_min"]
            - result["test_above_train_max"]
        )
        counts = [
            within,
            result["test_below_train_min"],
            result["test_above_train_max"],
        ]
        colors = ["#10b981", "#f59e0b", "#ef4444"]

        wedges, texts, autotexts = ax4.pie(
            counts, labels=categories, autopct="%1.1f%%", colors=colors, startangle=90
        )
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontsize(12)
            autotext.set_fontweight("bold")

        ax4.set_title(
            f"Test Data Range Analysis\n(relative to {self.dataset_names[0]})",
            fontsize=13,
            fontweight="bold",
        )

        # 5. Summary
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis("off")

        summary_text = f"""
SUMMARY STATISTICS
{'='*40}

Feature: {feature}
Type: Numerical

VALUE RANGES:
  • {self.dataset_names[0]}: [{result['train_min']:.2f}, {result['train_max']:.2f}]
  • {self.dataset_names[1]}: [{result['test_min']:.2f}, {result['test_max']:.2f}]

CENTRAL TENDENCY:
  • Mean change: {result['mean_change']:.2f} ({result['mean_pct_change']:+.1f}%)
  • {self.dataset_names[0]} mean: {result['train_mean']:.2f}
  • {self.dataset_names[1]} mean: {result['test_mean']:.2f}

OUT OF BOUNDS:
  • Below train min: {result['test_below_train_min']} ({result['test_below_train_min']/len(result['test_vals'])*100:.1f}%)
  • Above train max: {result['test_above_train_max']} ({result['test_above_train_max']/len(result['test_vals'])*100:.1f}%)

DRIFT METRICS:
  • KS statistic: {result['ks_stat']:.4f}
  • KS p-value: {result['ks_p']:.4f}

ASSESSMENT:
"""

        if result["out_of_bounds_pct"] > 10:
            summary_text += f"  🚨 CRITICAL - {result['out_of_bounds_pct']:.1f}%\n"
            summary_text += "  out of train range"
        elif result["out_of_bounds_pct"] > 5:
            summary_text += f"  ⚠️ WARNING - {result['out_of_bounds_pct']:.1f}%\n"
            summary_text += "  out of train range"
        elif result["ks_p"] < 0.01:
            summary_text += "  🚨 CRITICAL - Significant\n  distribution shift"
        elif result["ks_p"] < 0.05:
            summary_text += "  ⚠️ MODERATE - Distribution\n  differences detected"
        else:
            summary_text += "  ✓ STABLE - Similar distributions"

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
            f"Numerical Feature Analysis: {feature}\nTrain vs Test Comparison",
            fontsize=17,
            fontweight="bold",
        )

        return fig

    def create_executive_summary(self):
        """Create executive summary page."""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

        # 1. Overall Drift Summary
        ax1 = fig.add_subplot(gs[0, :])

        cat_critical = sum(
            1
            for f in self.categorical_features
            if self.results[f]["new_categories_count"] > 0
            and self.results[f]["new_cat_proportion"] > 0.1
        )
        cat_warning = sum(
            1
            for f in self.categorical_features
            if self.results[f]["drift_detected"]
            and not (
                self.results[f]["new_categories_count"] > 0
                and self.results[f]["new_cat_proportion"] > 0.1
            )
        )
        cat_stable = len(self.categorical_features) - cat_critical - cat_warning

        num_critical = sum(
            1
            for f in self.numerical_features
            if self.results[f]["out_of_bounds_pct"] > 10
        )
        num_warning = sum(
            1
            for f in self.numerical_features
            if self.results[f]["drift_detected"]
            and self.results[f]["out_of_bounds_pct"] <= 10
        )
        num_stable = len(self.numerical_features) - num_critical - num_warning

        categories = ["Critical", "Warning", "Stable"]
        cat_counts = [cat_critical, cat_warning, cat_stable]
        num_counts = [num_critical, num_warning, num_stable]

        x = np.arange(len(categories))
        width = 0.35

        ax1.bar(
            x - width / 2,
            cat_counts,
            width,
            label="Categorical",
            alpha=0.8,
            color="#3b82f6",
        )
        ax1.bar(
            x + width / 2,
            num_counts,
            width,
            label="Numerical",
            alpha=0.8,
            color="#10b981",
        )

        ax1.set_ylabel("Number of Features", fontsize=13, fontweight="bold")
        ax1.set_title("Feature Drift Status Summary", fontsize=16, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend(fontsize=12)
        ax1.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for i, (cat_val, num_val) in enumerate(zip(cat_counts, num_counts)):
            ax1.text(
                i - width / 2,
                cat_val,
                str(cat_val),
                ha="center",
                va="bottom",
                fontweight="bold",
            )
            ax1.text(
                i + width / 2,
                num_val,
                str(num_val),
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 2. Top Problematic Categorical Features
        ax2 = fig.add_subplot(gs[1, 0])

        cat_problems = [
            (
                f,
                self.results[f]["new_categories_count"],
                self.results[f]["new_cat_proportion"],
            )
            for f in self.categorical_features
            if self.results[f]["new_categories_count"] > 0
        ]
        cat_problems.sort(key=lambda x: x[2], reverse=True)
        cat_problems = cat_problems[:10]

        if cat_problems:
            feats, new_counts, proportions = zip(*cat_problems)
            colors = ["#ef4444" if p > 0.1 else "#f59e0b" for p in proportions]
            ax2.barh(
                range(len(feats)),
                [p * 100 for p in proportions],
                color=colors,
                alpha=0.8,
            )
            ax2.set_yticks(range(len(feats)))
            ax2.set_yticklabels(feats, fontsize=10)
            ax2.set_xlabel(
                "% Test Data with New Categories", fontsize=12, fontweight="bold"
            )
            ax2.set_title(
                "Top Categorical Features with New Categories",
                fontsize=13,
                fontweight="bold",
            )
            ax2.grid(axis="x", alpha=0.3)
        else:
            ax2.text(
                0.5,
                0.5,
                "No new categories detected\n✓ All categorical features stable",
                ha="center",
                va="center",
                fontsize=14,
                transform=ax2.transAxes,
            )
            ax2.axis("off")

        # 3. Top Problematic Numerical Features
        ax3 = fig.add_subplot(gs[1, 1])

        num_problems = [
            (f, self.results[f]["out_of_bounds_pct"]) for f in self.numerical_features
        ]
        num_problems.sort(key=lambda x: x[1], reverse=True)
        num_problems = num_problems[:10]

        if any(pct > 0 for _, pct in num_problems):
            feats, pcts = zip(*num_problems)
            colors = [
                "#ef4444" if p > 10 else "#f59e0b" if p > 5 else "#10b981" for p in pcts
            ]
            ax3.barh(range(len(feats)), pcts, color=colors, alpha=0.8)
            ax3.set_yticks(range(len(feats)))
            ax3.set_yticklabels(feats, fontsize=10)
            ax3.set_xlabel(
                "% Test Data Out of Train Range", fontsize=12, fontweight="bold"
            )
            ax3.set_title(
                "Top Numerical Features Out-of-Range", fontsize=13, fontweight="bold"
            )
            ax3.grid(axis="x", alpha=0.3)
        else:
            ax3.text(
                0.5,
                0.5,
                "No out-of-bounds data detected\n✓ All numerical features stable",
                ha="center",
                va="center",
                fontsize=14,
                transform=ax3.transAxes,
            )
            ax3.axis("off")

        # 4. Report Info & Summary Text
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis("off")

        critical_features = [
            f
            for f, res in self.results.items()
            if (
                res["type"] == "categorical"
                and res["new_categories_count"] > 0
                and res["new_cat_proportion"] > 0.1
            )
            or (res["type"] == "numerical" and res["out_of_bounds_pct"] > 10)
        ]

        warning_features = [
            f
            for f, res in self.results.items()
            if res["drift_detected"] and f not in critical_features
        ]

        summary_text = f"""
REPORT INFORMATION
{'='*60}
Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{self.dataset_names[0]} size: {len(self.train_df):,} rows
{self.dataset_names[1]} size: {len(self.test_df):,} rows
Total common features analyzed: {len(self.common_features)}
  • Categorical: {len(self.categorical_features)}
  • Numerical: {len(self.numerical_features)}

OVERALL ASSESSMENT
{'='*60}
Total features with drift: {len(critical_features) + len(warning_features)} / {len(self.common_features)}
  • 🚨 Critical Drift: {len(critical_features)} features
  • ⚠️ Warning / Moderate Drift: {len(warning_features)} features
  • ✓ Stable: {len(self.common_features) - len(critical_features) - len(warning_features)} features

RECOMMENDATIONS
{'='*60}
"""
        if critical_features:
            summary_text += "🚨 **CRITICAL ACTION REQUIRED**:\n"
            summary_text += f"   Features with critical drift ({len(critical_features)}) require immediate investigation.\n"
            summary_text += "   - **New categories** may cause model errors (e.g., in `LabelEncoder`).\n"
            summary_text += (
                "   - **Out-of-range numericals** can lead to poor extrapolation.\n"
            )
            summary_text += f"   - **Review features:** {', '.join(critical_features[:5])}{'...' if len(critical_features) > 5 else ''}\n\n"
        elif warning_features:
            summary_text += "⚠️ **REVIEW RECOMMENDED**:\n"
            summary_text += f"   Features with moderate drift ({len(warning_features)}) should be reviewed.\n"
            summary_text += "   - These distribution shifts may degrade model performance over time.\n"
            summary_text += "   - Consider model retraining or feature engineering.\n"
            summary_text += f"   - **Review features:** {', '.join(warning_features[:5])}{'...' if len(warning_features) > 5 else ''}\n\n"
        else:
            summary_text += "✓ **DATASET LOOKS STABLE**:\n"
            summary_text += (
                "   No significant drift detected. The test data appears consistent\n"
            )
            summary_text += (
                "   with the training data. Proceed with model deployment/evaluation."
            )

        ax4.text(
            0.05,
            0.95,
            summary_text,
            fontsize=12,
            verticalalignment="top",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="#f0f9ff", alpha=0.8),
        )

        plt.suptitle(
            f"Executive Summary: Train vs Test Drift Analysis",
            fontsize=20,
            fontweight="bold",
        )

        return fig

    def generate_report(self, filename="drift_analysis_report.pdf"):
        """
        Generate a comprehensive PDF report of the drift analysis.

        Parameters:
        -----------
        filename : str, optional
            The name of the output PDF file (default: "drift_analysis_report.pdf")
        """
        if not self.results:
            print("Analysis not yet run. Running complete analysis first...")
            self.run_complete_analysis()

        print(f"Generating PDF report: {filename} ...")

        with PdfPages(filename) as pdf:
            # 1. Executive Summary
            try:
                print("  - Creating Executive Summary...")
                summary_fig = self.create_executive_summary()
                pdf.savefig(summary_fig)
                plt.close(summary_fig)
            except Exception as e:
                print(f"    ERROR creating executive summary: {e}")
                plt.close()  # Close any dangling plot

            # 2. Categorical Features
            print(
                f"  - Adding {len(self.categorical_features)} categorical feature pages..."
            )
            for feature in sorted(self.categorical_features):
                try:
                    fig = self.create_categorical_viz(feature)
                    pdf.savefig(fig)
                    plt.close(fig)
                except Exception as e:
                    print(f"    ERROR creating plot for '{feature}': {e}")
                    plt.close()  # Close any dangling plot

            # 3. Numerical Features
            print(
                f"  - Adding {len(self.numerical_features)} numerical feature pages..."
            )
            for feature in sorted(self.numerical_features):
                try:
                    fig = self.create_numerical_viz(feature)
                    pdf.savefig(fig)
                    plt.close(fig)
                except Exception as e:
                    print(f"    ERROR creating plot for '{feature}': {e}")
                    plt.close()  # Close any dangling plot

        print(f"\n{'='*80}")
        print(f"✓ Report successfully generated: {filename}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    # 1. Create Sample Data

    # Training Data
    train_data = {
        "age": np.random.normal(35, 10, 1000),
        "salary": np.random.lognormal(11, 0.5, 1000),
        "department": np.random.choice(
            ["Sales", "Engineering", "HR", "Marketing"], 1000, p=[0.4, 0.3, 0.1, 0.2]
        ),
        "region": np.random.choice(["North", "South", "East"], 1000),
        "tenure": np.random.randint(1, 10, 1000),
    }
    train_df = pd.DataFrame(train_data)

    # Test Data with Drift
    test_data = {
        # 'age' has a slight shift
        "age": np.random.normal(38, 12, 500),
        # 'salary' has a larger shift and out-of-bounds values
        "salary": np.random.lognormal(11.2, 0.6, 500),
        # 'department' has a new category 'Support' and 'HR' is missing
        "department": np.random.choice(
            ["Sales", "Engineering", "Support", "Marketing"],
            500,
            p=[0.3, 0.2, 0.2, 0.3],
        ),
        # 'region' is stable
        "region": np.random.choice(["North", "South", "East"], 500),
        # 'tenure' has out-of-bounds values (new max)
        "tenure": np.random.randint(1, 15, 500),
    }
    test_df = pd.DataFrame(test_data)

    # Add some nulls
    train_df.loc[train_df.sample(frac=0.05).index, "age"] = np.nan
    test_df.loc[test_df.sample(frac=0.07).index, "salary"] = np.nan

    print("Sample data created.")
    print("--- Train Data Info ---")
    train_df.info()
    print("\n--- Test Data Info ---")
    test_df.info()

    # 2. Initialize and Run Analysis
    analyzer = TrainTestDriftAnalysis(
        train_df, test_df, dataset_names=("Training Set", "Production Data")
    )

    # 3. Run the full analysis (this is also called by generate_report if not run)
    analyzer.run_complete_analysis()

    # 4. Generate the PDF Report
    analyzer.generate_report(filename="Sample_Drift_Report.pdf")

    # 5. Access specific results
    if "department" in analyzer.results:
        print("\n--- Specific results for 'department' ---")
        dept_results = analyzer.results["department"]
        print(f"  Drift detected: {dept_results['drift_detected']}")
        print(f"  New categories: {dept_results['new_categories']}")
        print(f"  TVD: {dept_results['tvd']:.4f}")

    if "age" in analyzer.results:
        print("\n--- Specific results for 'age' ---")
        age_results = analyzer.results["age"]
        print(f"  Drift detected: {age_results['drift_detected']}")
        print(f"  KS p-value: {age_results['ks_p']:.4f}")
        print(f"  % out of bounds: {age_results['out_of_bounds_pct']:.2f}%")
