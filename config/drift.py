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

from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class MultiMonthDriftAnalysis:
    """
    Comprehensive multi-month drift analysis with PDF report generation.
    Analyzes all months sequentially and creates detailed visualizations for each variable.
    """

    def __init__(self, df, month_col):
        """
        Initialize multi-month drift analyzer.

        Parameters:
        -----------
        df : pd.DataFrame
            Your dataset with multiple months
        month_col : str
            Name of the month/date column
        """
        self.df = df.copy()
        self.month_col = month_col

        # Get sorted months
        self.months = sorted(df[month_col].unique())
        print(f"\n{'='*80}")
        print(f"MULTI-MONTH DRIFT ANALYSIS")
        print(f"{'='*80}")
        print(f"Total months: {len(self.months)}")
        print(f"Months: {', '.join(map(str, self.months))}")
        print(f"Total rows: {len(df):,}")

        # Identify feature types
        self.all_features = [col for col in df.columns if col != month_col]
        self.categorical_features = []
        self.numerical_features = []

        for feat in self.all_features:
            if df[feat].dtype == "object" or df[feat].nunique() < 20:
                self.categorical_features.append(feat)
            else:
                self.numerical_features.append(feat)

        print(f"\nCategorical features: {len(self.categorical_features)}")
        print(f"Numerical features: {len(self.numerical_features)}")
        print(f"{'='*80}\n")

        # Store results for all month pairs
        self.monthly_results = {}

    def analyze_categorical_month_over_month(self, feature):
        """Analyze a categorical feature across all months."""
        results = []

        for i in range(len(self.months) - 1):
            month1 = self.months[i]
            month2 = self.months[i + 1]

            df1 = self.df[self.df[self.month_col] == month1]
            df2 = self.df[self.df[self.month_col] == month2]

            dist1 = df1[feature].value_counts(normalize=True)
            dist2 = df2[feature].value_counts(normalize=True)

            # Calculate TVD
            all_cats = set(dist1.index) | set(dist2.index)
            tvd = 0.5 * sum(
                [abs(dist1.get(cat, 0) - dist2.get(cat, 0)) for cat in all_cats]
            )

            # Calculate JSD
            all_categories = sorted(all_cats)
            p = np.array([dist1.get(cat, 1e-10) for cat in all_categories])
            q = np.array([dist2.get(cat, 1e-10) for cat in all_categories])
            p = p / p.sum()
            q = q / q.sum()
            jsd = jensenshannon(p, q) ** 2

            # Chi-square test
            observed = []
            for cat in all_cats:
                observed.append(
                    [(dist1.get(cat, 0) * len(df1)), (dist2.get(cat, 0) * len(df2))]
                )
            observed = np.array(observed)

            try:
                chi2, p_value, _, _ = stats.chi2_contingency(observed.T)
            except:
                chi2, p_value = 0, 1.0

            results.append(
                {
                    "month_pair": f"{month1}→{month2}",
                    "month1": month1,
                    "month2": month2,
                    "tvd": tvd,
                    "jsd": jsd,
                    "chi2": chi2,
                    "p_value": p_value,
                    "drift_detected": tvd > 0.1 or p_value < 0.05,
                    "dist1": dist1,
                    "dist2": dist2,
                }
            )

        return results

    def analyze_numerical_month_over_month(self, feature):
        """Analyze a numerical feature across all months."""
        results = []

        for i in range(len(self.months) - 1):
            month1 = self.months[i]
            month2 = self.months[i + 1]

            vals1 = self.df[self.df[self.month_col] == month1][feature].dropna()
            vals2 = self.df[self.df[self.month_col] == month2][feature].dropna()

            # Statistical tests
            ks_stat, ks_p = stats.ks_2samp(vals1, vals2)
            t_stat, t_p = stats.ttest_ind(vals1, vals2)

            # Statistics
            mean1, mean2 = vals1.mean(), vals2.mean()
            median1, median2 = vals1.median(), vals2.median()
            std1, std2 = vals1.std(), vals2.std()

            mean_change = mean2 - mean1
            mean_pct_change = (mean_change / (mean1 + 1e-10)) * 100

            results.append(
                {
                    "month_pair": f"{month1}→{month2}",
                    "month1": month1,
                    "month2": month2,
                    "mean1": mean1,
                    "mean2": mean2,
                    "median1": median1,
                    "median2": median2,
                    "std1": std1,
                    "std2": std2,
                    "mean_change": mean_change,
                    "mean_pct_change": mean_pct_change,
                    "ks_stat": ks_stat,
                    "ks_p": ks_p,
                    "t_stat": t_stat,
                    "t_p": t_p,
                    "drift_detected": ks_p < 0.05 or abs(mean_pct_change) > 10,
                    "vals1": vals1,
                    "vals2": vals2,
                }
            )

        return results

    def run_complete_analysis(self):
        """Run analysis on all features across all months."""
        print("Analyzing all features month-over-month...")
        print(f"{'='*80}\n")

        # Analyze categorical features
        print(f"📊 Analyzing {len(self.categorical_features)} categorical features...")
        for feat in self.categorical_features:
            self.monthly_results[feat] = {
                "type": "categorical",
                "results": self.analyze_categorical_month_over_month(feat),
            }
            drift_count = sum(
                [r["drift_detected"] for r in self.monthly_results[feat]["results"]]
            )
            status = (
                f"🚨 {drift_count}/{len(self.months)-1} months drifting"
                if drift_count > 0
                else "✓ Stable"
            )
            print(f"   {feat}: {status}")

        # Analyze numerical features
        print(f"\n📈 Analyzing {len(self.numerical_features)} numerical features...")
        for feat in self.numerical_features:
            self.monthly_results[feat] = {
                "type": "numerical",
                "results": self.analyze_numerical_month_over_month(feat),
            }
            drift_count = sum(
                [r["drift_detected"] for r in self.monthly_results[feat]["results"]]
            )
            status = (
                f"🚨 {drift_count}/{len(self.months)-1} months drifting"
                if drift_count > 0
                else "✓ Stable"
            )
            print(f"   {feat}: {status}")

        print(f"\n{'='*80}")
        print("✓ Analysis complete!")
        print(f"{'='*80}\n")

    def create_categorical_viz(self, feature):
        """Create comprehensive visualization for a categorical feature."""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

        results = self.monthly_results[feature]["results"]

        # 1. TVD Over Time (Line Chart)
        ax1 = fig.add_subplot(gs[0, :])
        month_pairs = [r["month_pair"] for r in results]
        tvds = [r["tvd"] for r in results]

        ax1.plot(
            month_pairs, tvds, marker="o", linewidth=3, markersize=10, color="#ef4444"
        )
        ax1.axhline(
            0.1,
            color="orange",
            linestyle="--",
            alpha=0.7,
            label="Moderate Threshold",
            linewidth=2,
        )
        ax1.axhline(
            0.2,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="High Threshold",
            linewidth=2,
        )
        ax1.fill_between(range(len(tvds)), tvds, alpha=0.3, color="#ef4444")
        ax1.set_xlabel("Month Transition", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Total Variation Distance", fontsize=13, fontweight="bold")
        ax1.set_title(
            f"{feature}: Drift Magnitude Over Time", fontsize=15, fontweight="bold"
        )
        ax1.tick_params(axis="x", rotation=45)
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3)

        # 2. Distribution Evolution (Stacked Area)
        ax2 = fig.add_subplot(gs[1, :])

        # Get top categories across all months
        all_dists = []
        for month in self.months:
            dist = self.df[self.df[self.month_col] == month][feature].value_counts(
                normalize=True
            )
            all_dists.append(dist)

        all_cats = set()
        for dist in all_dists:
            all_cats.update(dist.index)

        # Keep top 10 categories
        total_counts = pd.Series(0.0, index=all_cats)
        for dist in all_dists:
            total_counts = total_counts.add(dist, fill_value=0)
        top_cats = total_counts.nlargest(10).index.tolist()

        # Create matrix for stacked area
        matrix = []
        for month in self.months:
            dist = self.df[self.df[self.month_col] == month][feature].value_counts(
                normalize=True
            )
            row = [dist.get(cat, 0) * 100 for cat in top_cats]
            matrix.append(row)

        matrix = np.array(matrix).T
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_cats)))

        ax2.stackplot(
            range(len(self.months)), matrix, labels=top_cats, colors=colors, alpha=0.8
        )
        ax2.set_xlabel("Month", fontsize=13, fontweight="bold")
        ax2.set_ylabel("Percentage (%)", fontsize=13, fontweight="bold")
        ax2.set_title("Category Distribution Evolution", fontsize=15, fontweight="bold")
        ax2.set_xticks(range(len(self.months)))
        ax2.set_xticklabels(self.months, rotation=45)
        ax2.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=9)
        ax2.grid(alpha=0.3)

        # 3. Drift Detection Heatmap
        ax3 = fig.add_subplot(gs[2, 0])

        drift_matrix = np.array([[1 if r["drift_detected"] else 0 for r in results]])
        im = ax3.imshow(drift_matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
        ax3.set_yticks([0])
        ax3.set_yticklabels(["Drift"])
        ax3.set_xticks(range(len(month_pairs)))
        ax3.set_xticklabels(month_pairs, rotation=45, ha="right", fontsize=9)
        ax3.set_title("Drift Detection Timeline", fontsize=13, fontweight="bold")

        # Add text annotations
        for i in range(len(month_pairs)):
            text = "✓" if results[i]["drift_detected"] else "○"
            ax3.text(
                i,
                0,
                text,
                ha="center",
                va="center",
                fontsize=16,
                color="white" if results[i]["drift_detected"] else "black",
                fontweight="bold",
            )

        # 4. Summary Statistics
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.axis("off")

        total_drifts = sum([r["drift_detected"] for r in results])
        avg_tvd = np.mean([r["tvd"] for r in results])
        max_tvd = max([r["tvd"] for r in results])
        max_drift_transition = results[[r["tvd"] for r in results].index(max_tvd)][
            "month_pair"
        ]

        unique_cats_start = len(
            self.df[self.df[self.month_col] == self.months[0]][feature].unique()
        )
        unique_cats_end = len(
            self.df[self.df[self.month_col] == self.months[-1]][feature].unique()
        )

        summary_text = f"""
SUMMARY STATISTICS
{'='*40}

Feature: {feature}
Type: Categorical

DRIFT METRICS:
  • Drifting Months: {total_drifts}/{len(results)}
  • Average TVD: {avg_tvd:.4f}
  • Maximum TVD: {max_tvd:.4f}
  • Worst Transition: {max_drift_transition}

CATEGORY CHANGES:
  • Start ({self.months[0]}): {unique_cats_start} categories
  • End ({self.months[-1]}): {unique_cats_end} categories
  • Net Change: {unique_cats_end - unique_cats_start:+d} categories

ASSESSMENT:
"""

        if total_drifts > len(results) * 0.7:
            summary_text += "  🚨 CRITICAL - Consistent drift\n  across most months"
        elif total_drifts > len(results) * 0.4:
            summary_text += "  ⚠️ SIGNIFICANT - Periodic drift\n  observed"
        else:
            summary_text += "  ✓ STABLE - Minor variations only"

        ax4.text(
            0.1,
            0.9,
            summary_text,
            fontsize=11,
            verticalalignment="top",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.suptitle(
            f"Categorical Feature Analysis: {feature}\nMonth-over-Month Drift Story",
            fontsize=17,
            fontweight="bold",
        )

        return fig

    def create_numerical_viz(self, feature):
        """Create comprehensive visualization for a numerical feature."""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

        results = self.monthly_results[feature]["results"]

        # 1. Mean/Median Evolution
        ax1 = fig.add_subplot(gs[0, :])

        means = [
            self.df[self.df[self.month_col] == m][feature].mean() for m in self.months
        ]
        medians = [
            self.df[self.df[self.month_col] == m][feature].median() for m in self.months
        ]

        ax1.plot(
            self.months,
            means,
            marker="o",
            linewidth=3,
            markersize=10,
            label="Mean",
            color="#3b82f6",
        )
        ax1.plot(
            self.months,
            medians,
            marker="s",
            linewidth=3,
            markersize=10,
            label="Median",
            color="#10b981",
        )
        ax1.fill_between(range(len(means)), means, alpha=0.2, color="#3b82f6")
        ax1.set_xlabel("Month", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Value", fontsize=13, fontweight="bold")
        ax1.set_title(
            f"{feature}: Central Tendency Evolution", fontsize=15, fontweight="bold"
        )
        ax1.tick_params(axis="x", rotation=45)
        ax1.legend(fontsize=12)
        ax1.grid(alpha=0.3)

        # 2. Distribution Evolution (Violin Plot)
        ax2 = fig.add_subplot(gs[1, 0])

        plot_data = []
        plot_labels = []
        for month in self.months:
            vals = self.df[self.df[self.month_col] == month][feature].dropna()
            plot_data.append(vals)
            plot_labels.append(str(month))

        parts = ax2.violinplot(
            plot_data,
            positions=range(len(self.months)),
            showmeans=True,
            showmedians=True,
        )
        for pc in parts["bodies"]:
            pc.set_alpha(0.7)
        ax2.set_xticks(range(len(self.months)))
        ax2.set_xticklabels(plot_labels, rotation=45)
        ax2.set_xlabel("Month", fontsize=12, fontweight="bold")
        ax2.set_ylabel(feature, fontsize=12, fontweight="bold")
        ax2.set_title("Distribution Shape Evolution", fontsize=13, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)

        # 3. Percentage Change Heatmap
        ax3 = fig.add_subplot(gs[1, 1])

        pct_changes = [r["mean_pct_change"] for r in results]
        month_pairs = [r["month_pair"] for r in results]

        colors_bar = ["#10b981" if x > 0 else "#ef4444" for x in pct_changes]
        ax3.barh(range(len(pct_changes)), pct_changes, color=colors_bar, alpha=0.8)
        ax3.set_yticks(range(len(pct_changes)))
        ax3.set_yticklabels(month_pairs, fontsize=10)
        ax3.axvline(0, color="black", linewidth=2)
        ax3.axvline(-10, color="orange", linestyle="--", alpha=0.5)
        ax3.axvline(10, color="orange", linestyle="--", alpha=0.5)
        ax3.set_xlabel("Mean % Change", fontsize=12, fontweight="bold")
        ax3.set_title("Month-over-Month % Changes", fontsize=13, fontweight="bold")
        ax3.grid(axis="x", alpha=0.3)

        # 4. Drift Detection Timeline
        ax4 = fig.add_subplot(gs[2, 0])

        drift_matrix = np.array([[1 if r["drift_detected"] else 0 for r in results]])
        im = ax4.imshow(drift_matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
        ax4.set_yticks([0])
        ax4.set_yticklabels(["Drift"])
        ax4.set_xticks(range(len(month_pairs)))
        ax4.set_xticklabels(month_pairs, rotation=45, ha="right", fontsize=9)
        ax4.set_title("Drift Detection Timeline", fontsize=13, fontweight="bold")

        for i in range(len(month_pairs)):
            text = "✓" if results[i]["drift_detected"] else "○"
            ax4.text(
                i,
                0,
                text,
                ha="center",
                va="center",
                fontsize=16,
                color="white" if results[i]["drift_detected"] else "black",
                fontweight="bold",
            )

        # 5. Summary Statistics
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis("off")

        total_drifts = sum([r["drift_detected"] for r in results])
        avg_pct_change = np.mean([abs(r["mean_pct_change"]) for r in results])
        max_pct_change = max([abs(r["mean_pct_change"]) for r in results])
        max_drift_idx = [abs(r["mean_pct_change"]) for r in results].index(
            max_pct_change
        )
        max_drift_transition = results[max_drift_idx]["month_pair"]

        start_mean = self.df[self.df[self.month_col] == self.months[0]][feature].mean()
        end_mean = self.df[self.df[self.month_col] == self.months[-1]][feature].mean()
        total_change = ((end_mean - start_mean) / start_mean) * 100

        summary_text = f"""
SUMMARY STATISTICS
{'='*40}

Feature: {feature}
Type: Numerical

DRIFT METRICS:
  • Drifting Months: {total_drifts}/{len(results)}
  • Avg |% Change|: {avg_pct_change:.2f}%
  • Max |% Change|: {max_pct_change:.2f}%
  • Worst Transition: {max_drift_transition}

VALUE CHANGES:
  • Start ({self.months[0]}): {start_mean:.2f}
  • End ({self.months[-1]}): {end_mean:.2f}
  • Total Change: {total_change:+.2f}%

ASSESSMENT:
"""

        if total_drifts > len(results) * 0.7:
            summary_text += "  🚨 CRITICAL - Consistent drift\n  throughout period"
        elif total_drifts > len(results) * 0.4:
            summary_text += "  ⚠️ SIGNIFICANT - Multiple drift\n  periods detected"
        else:
            summary_text += "  ✓ STABLE - Minor fluctuations"

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
            f"Numerical Feature Analysis: {feature}\nMonth-over-Month Drift Story",
            fontsize=17,
            fontweight="bold",
        )

        return fig


def generate_pdf_report(analyzer, filename=None):
    """
    Generate a complete multi-month drift PDF report for the given analyzer.

    Parameters
    ----------
    analyzer : MultiMonthDriftAnalysis
        An instance of the MultiMonthDriftAnalysis class after running analysis.
    filename : str, optional
        Output filename for the PDF. If not provided, it will include a timestamp.

    Returns
    -------
    str
        Path to the generated PDF file.
    """
    if filename is None:
        filename = f"multi_month_drift_report_{datetime.now():%Y%m%d_%H%M}.pdf"

    print(f"\n📄 Generating comprehensive drift report: {filename}")

    with PdfPages(filename) as pdf:
        # Executive summary
        try:
            fig_summary = analyzer.create_executive_summary()
            pdf.savefig(fig_summary)
            plt.close(fig_summary)
            print("✓ Added Executive Summary")
        except Exception as e:
            print(f"⚠️  Skipped executive summary (error: {e})")

        # Categorical features
        for feat in analyzer.categorical_features:
            try:
                fig = analyzer.create_categorical_viz(feat)
                pdf.savefig(fig)
                plt.close(fig)
                print(f"✓ Added categorical feature: {feat}")
            except Exception as e:
                print(f"⚠️  Skipped {feat} (error: {e})")

        # Numerical features
        for feat in analyzer.numerical_features:
            try:
                fig = analyzer.create_numerical_viz(feat)
                pdf.savefig(fig)
                plt.close(fig)
                print(f"✓ Added numerical feature: {feat}")
            except Exception as e:
                print(f"⚠️  Skipped {feat} (error: {e})")

    print(f"\n✅ Multi-Month Drift Report generated successfully: {filename}")
    return filename

    def create_executive_summary(self):
        """Create executive summary page."""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

        # 1. Overall Drift Prevalence
        ax1 = fig.add_subplot(gs[0, :])

        cat_drift_counts = []
        num_drift_counts = []

        for i in range(len(self.months) - 1):
            cat_drifts = sum(
                [
                    self.monthly_results[f]["results"][i]["drift_detected"]
                    for f in self.categorical_features
                ]
            )
            num_drifts = sum(
                [
                    self.monthly_results[f]["results"][i]["drift_detected"]
                    for f in self.numerical_features
                ]
            )
            cat_drift_counts.append(cat_drifts)
            num_drift_counts.append(num_drifts)

        month_pairs = [
            f"{self.months[i]}→{self.months[i+1]}" for i in range(len(self.months) - 1)
        ]
        x = np.arange(len(month_pairs))
        width = 0.35

        ax1.bar(
            x - width / 2,
            cat_drift_counts,
            width,
            label="Categorical",
            alpha=0.8,
            color="#3b82f6",
        )
        ax1.bar(
            x + width / 2,
            num_drift_counts,
            width,
            label="Numerical",
            alpha=0.8,
            color="#10b981",
        )
        ax1.set_xlabel("Month Transition", fontsize=13, fontweight="bold")
        ax1.set_ylabel("# Features Drifting", fontsize=13, fontweight="bold")
        ax1.set_title("Drift Prevalence Over Time", fontsize=16, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(month_pairs, rotation=45, ha="right")
        ax1.legend(fontsize=12)
        ax1.grid(axis="y", alpha=0.3)

        # 2. Feature Stability Ranking (Categorical)
        ax2 = fig.add_subplot(gs[1, 0])

        cat_stability = {}
        for feat in self.categorical_features:
            drift_count = sum(
                [r["drift_detected"] for r in self.monthly_results[feat]["results"]]
            )
            cat_stability[feat] = drift_count

        cat_sorted = sorted(cat_stability.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]
        if cat_sorted:
            feats, counts = zip(*cat_sorted)
            colors = [
                (
                    "#ef4444"
                    if c > len(self.months) * 0.5
                    else "#f59e0b" if c > len(self.months) * 0.25 else "#10b981"
                )
                for c in counts
            ]
            ax2.barh(range(len(feats)), counts, color=colors, alpha=0.8)
            ax2.set_yticks(range(len(feats)))
            ax2.set_yticklabels(feats, fontsize=10)
            ax2.set_xlabel("# Months Drifting", fontsize=12, fontweight="bold")
            ax2.set_title(
                "Top Drifting Categorical Features", fontsize=13, fontweight="bold"
            )
            ax2.grid(axis="x", alpha=0.3)

        # 3. Feature Stability Ranking (Numerical)
        ax3 = fig.add_subplot(gs[1, 1])

        num_stability = {}
        for feat in self.numerical_features:
            drift_count = sum(
                [r["drift_detected"] for r in self.monthly_results[feat]["results"]]
            )
            num_stability[feat] = drift_count

        num_sorted = sorted(num_stability.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]
        if num_sorted:
            feats, counts = zip(*num_sorted)
            colors = [
                (
                    "#ef4444"
                    if c > len(self.months) * 0.5
                    else "#f59e0b" if c > len(self.months) * 0.25 else "#10b981"
                )
                for c in counts
            ]
            ax3.barh(range(len(feats)), counts, color=colors, alpha=0.8)
            ax3.set_yticks(range(len(feats)))
            ax3.set_yticklabels(feats, fontsize=10)
            ax3.set_xlabel("# Months Drifting", fontsize=12, fontweight="bold")
            ax3.set_title(
                "Top Drifting Numerical Features", fontsize=13, fontweight="bold"
            )
            ax3.grid(axis="x", alpha=0.3)

        # 4. Overall Summary Text
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis("off")

        total_features = len(self.all_features)
        total_transitions = len(self.months) - 1

        # Calculate overall drift statistics
        highly_unstable = sum(
            [
                1
                for f in self.all_features
                if sum(
                    [r["drift_detected"] for r in self.monthly_results[f]["results"]]
                )
                > total_transitions * 0.7
            ]
        )
        moderately_unstable = sum(
            [
                1
                for f in self.all_features
                if total_transitions * 0.3
                < sum([r["drift_detected"] for r in self.monthly_results[f]["results"]])
                <= total_transitions * 0.7
            ]
        )
        stable = total_features - highly_unstable - moderately_unstable

        summary_text = f"""
{'='*80}
                    EXECUTIVE SUMMARY - MULTI-MONTH DRIFT ANALYSIS
{'='*80}

Analysis Period: {self.months[0]} to {self.months[-1]} ({len(self.months)} months)
Total Features Analyzed: {total_features} ({len(self.categorical_features)} categorical, {len(self.numerical_features)} numerical)
Total Month Transitions: {total_transitions}

FEATURE STABILITY ASSESSMENT:
{'─'*80}
  🚨 Highly Unstable (>70% months drifting):    {highly_unstable} features ({highly_unstable/total_features*100:.1f}%)
  ⚠️  Moderately Unstable (30-70% drifting):     {moderately_unstable} features ({moderately_unstable/total_features*100:.1f}%)
  ✓ Stable (<30% drifting):                      {stable} features ({stable/total_features*100:.1f}%)

CRITICAL INSIGHTS:
{'─'*80}
"""

        if highly_unstable > total_features * 0.5:
            summary_text += (
                "  🚨 CRITICAL ALERT: Majority of features showing persistent drift\n"
            )
            summary_text += "     → Immediate investigation required\n"
            summary_text += "     → Potential data quality or collection issues\n"
            summary_text += "     → Model retraining urgently needed\n"
        elif highly_unstable > total_features * 0.25:
            summary_text += "  ⚠️  SIGNIFICANT CONCERN: Many features unstable\n"
            summary_text += "     → Review data pipeline and feature engineering\n"
            summary_text += "     → Schedule model retraining\n"
            summary_text += "     → Monitor model performance closely\n"
        else:
            summary_text += "  ✓ MANAGEABLE SITUATION: Most features stable\n"
            summary_text += "     → Continue monitoring\n"
            summary_text += "     → Focus on highly unstable features\n"
            summary_text += "     → Standard maintenance procedures\n"

        summary_text += f"\n{'='*80}"
        summary_text += (
            f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        summary_text += f"\n{'='*80}"

        ax4.text(
            0.02,
            0.98,
            summary_text,
            fontsize=11,
            verticalalignment="top",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.suptitle(
            f"Executive Summary: Multi-Month Drift Analysis",
            fontsize=18,
            fontweight="bold",
        )
        plt.tight_layout()
        return fig


    def get_stability_summary(analyzer):
    """
    Return lists of stable, moderately drifting, and highly drifting variables.
    """
    stable_vars = []
    moderate_vars = []
    high_drift_vars = []

    total_transitions = len(analyzer.months) - 1

    for feat in analyzer.all_features:
        drift_count = sum(r['drift_detected'] for r in analyzer.monthly_results[feat]['results'])
        drift_ratio = drift_count / total_transitions

        if drift_ratio <= 0.3:
            stable_vars.append(feat)
        elif drift_ratio <= 0.7:
            moderate_vars.append(feat)
        else:
            high_drift_vars.append(feat)

    print("\n=== FEATURE STABILITY SUMMARY ===")
    print(f"Stable features ({len(stable_vars)}): {stable_vars}")
    print(f"Moderate drift ({len(moderate_vars)}): {moderate_vars}")
    print(f"High drift ({len(high_drift_vars)}): {high_drift_vars}")
    print("=================================\n")

    return {
        "stable": stable_vars,
        "moderate": moderate_vars,
        "high_drift": high_drift_vars
    }
