from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import yaml
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier


class HyperparameterTuner:
    def __init__(self, model, param_grid: Dict[str, Any], method: str = "optuna"):
        """
        Initialize hyperparameter tuner

        Args:
            model: Base model to tune
            param_grid: Parameter grid to search
            method: Tuning method ('grid', 'random', 'optuna')
        """
        self.model = model
        self.param_grid = param_grid
        self.method = method
        self.best_params = None
        self.best_score = None
        self.best_model = None

    def objective(self, trial):
        """Optuna objective function"""
        params = {}
        for param, values in self.param_grid.items():
            if isinstance(values, list):
                if isinstance(values[0], int):
                    params[param] = trial.suggest_int(param, min(values), max(values))
                elif isinstance(values[0], float):
                    params[param] = trial.suggest_float(param, min(values), max(values))
                else:
                    params[param] = trial.suggest_categorical(param, values)

        model = self.model.set_params(**params)
        model.fit(self.X_train, self.y_train)
        return -f1_score(
            self.y_val, model.predict(self.X_val)
        )  # Negative because optuna minimizes

    def tune(self, X_train, y_train, X_val, y_val, **kwargs):
        """
        Tune hyperparameters using specified method
        """
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val

        if self.method == "grid":
            tuner = GridSearchCV(
                self.model,
                self.param_grid,
                cv=kwargs.get("cv", 5),
                scoring=kwargs.get("scoring", "f1"),
                n_jobs=kwargs.get("n_jobs", -1),
            )
            tuner.fit(X_train, y_train)
            self.best_params = tuner.best_params_
            self.best_score = tuner.best_score_
            self.best_model = tuner.best_estimator_

        elif self.method == "random":
            tuner = RandomizedSearchCV(
                self.model,
                self.param_grid,
                n_iter=kwargs.get("n_iter", 100),
                cv=kwargs.get("cv", 5),
                scoring=kwargs.get("scoring", "f1"),
                n_jobs=kwargs.get("n_jobs", -1),
            )
            tuner.fit(X_train, y_train)
            self.best_params = tuner.best_params_
            self.best_score = tuner.best_score_
            self.best_model = tuner.best_estimator_

        elif self.method == "optuna":
            study = optuna.create_study(direction="minimize")
            study.optimize(self.objective, n_trials=kwargs.get("n_trials", 100))
            self.best_params = study.best_params
            self.best_score = -study.best_value  # Convert back to positive
            self.best_model = self.model.set_params(**self.best_params)
            self.best_model.fit(X_train, y_train)

        return self.best_model, self.best_params, self.best_score


class AutoMLTrainer:
    def __init__(self, config_path: str):
        """Initialize with path to YAML config file"""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.models = self._initialize_models()
        self.results = {}

    def _initialize_models(self) -> Dict:
        """Initialize model dictionary based on YAML config"""
        model_mapping = {
            "random_forest": RandomForestClassifier,
            "xgboost": XGBClassifier,
            "catboost": CatBoostClassifier,
            "lightgbm": LGBMClassifier,
        }

        models = {}
        for model_name, model_config in self.config["models"].items():
            if model_name in model_mapping:
                models[model_name] = {
                    "model": model_mapping[model_name](),
                    "params": model_config["hyperparameters"],
                }
        return models

    def train_and_evaluate(self, X_train, X_test, X_val, y_train, y_test, y_val):
        """Train models with hyperparameter tuning and evaluate performance"""
        tuning_config = self.config.get("tuning", {})
        method = tuning_config.get("method", "optuna")

        for name, model_info in self.models.items():
            print(f"\nTraining {name}...")

            # Initialize and run hyperparameter tuning
            tuner = HyperparameterTuner(
                model_info["model"], model_info["params"], method=method
            )

            best_model, best_params, best_score = tuner.tune(
                X_train, y_train, X_val, y_val, **tuning_config.get("parameters", {})
            )

            # Generate predictions
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]

            # Store results
            self.results[name] = {
                "model": best_model,
                "best_params": best_params,
                "best_score": best_score,
                "f1_score": f1_score(y_test, y_pred),
                "brier_score": brier_score_loss(y_test, y_pred_proba),
                "predictions": y_pred,
                "probabilities": y_pred_proba,
            }

    def calculate_ks_stat(self, y_true, y_pred_proba) -> Tuple:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        ks_statistic = max(abs(tpr - fpr))
        return fpr, tpr, ks_statistic

    def calculate_lift(self, y_true, y_pred_proba, bins=10) -> pd.DataFrame:
        df = pd.DataFrame({"target": y_true, "proba": y_pred_proba})
        df["decile"] = pd.qcut(df["proba"], bins, labels=False)
        lift_data = df.groupby("decile")["target"].agg(["count", "mean"])
        lift_data["lift"] = lift_data["mean"] / df["target"].mean()
        return lift_data

    def plot_metrics(self, y_test):
        """Generate comprehensive performance visualization"""
        plt.style.use("seaborn")
        fig = plt.figure(figsize=(20, 20))
        gs = fig.add_gridspec(3, 2)

        # F1 Scores
        ax1 = fig.add_subplot(gs[0, 0])
        f1_scores = {
            name: results["f1_score"] for name, results in self.results.items()
        }
        sns.barplot(x=list(f1_scores.keys()), y=list(f1_scores.values()), ax=ax1)
        ax1.set_title("F1 Scores Comparison")
        ax1.set_ylim(0, 1)
        plt.xticks(rotation=45)

        # ROC Curves
        ax2 = fig.add_subplot(gs[0, 1])
        for name, results in self.results.items():
            fpr, tpr, _ = roc_curve(y_test, results["probabilities"])
            roc_auc = auc(fpr, tpr)
            ax2.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

        ax2.plot([0, 1], [0, 1], "k--")
        ax2.set_title("ROC Curves")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.legend()

        # KS Charts
        ax3 = fig.add_subplot(gs[1, 0])
        for name, results in self.results.items():
            fpr, tpr, ks_stat = self.calculate_ks_stat(y_test, results["probabilities"])
            ax3.plot(fpr, tpr, label=f"{name} (KS = {ks_stat:.2f})")
            ax3.plot(fpr, fpr, "k--")
        ax3.set_title("KS Charts")
        ax3.set_xlabel("False Positive Rate")
        ax3.set_ylabel("True Positive Rate")
        ax3.legend()

        # Lift Charts
        ax4 = fig.add_subplot(gs[1, 1])
        for name, results in self.results.items():
            lift_data = self.calculate_lift(y_test, results["probabilities"])
            ax4.plot(range(1, 11), lift_data["lift"], marker="o", label=name)
        ax4.set_title("Lift Charts")
        ax4.set_xlabel("Decile")
        ax4.set_ylabel("Lift")
        ax4.legend()

        # Precision-Recall Curves
        ax5 = fig.add_subplot(gs[2, 0])
        for name, results in self.results.items():
            precision, recall, _ = precision_recall_curve(
                y_test, results["probabilities"]
            )
            ax5.plot(recall, precision, label=name)
        ax5.set_title("Precision-Recall Curves")
        ax5.set_xlabel("Recall")
        ax5.set_ylabel("Precision")
        ax5.legend()

        # Brier Scores
        ax6 = fig.add_subplot(gs[2, 1])
        brier_scores = {
            name: results["brier_score"] for name, results in self.results.items()
        }
        sns.barplot(x=list(brier_scores.keys()), y=list(brier_scores.values()), ax=ax6)
        ax6.set_title("Brier Scores Comparison")
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def generate_summary_report(self) -> pd.DataFrame:
        """Generate comprehensive performance summary"""
        summary = pd.DataFrame()

        for name, results in self.results.items():
            model_summary = pd.Series(
                {
                    "Best Parameters": str(results["best_params"]),
                    "Best Validation Score": results["best_score"],
                    "F1 Score": results["f1_score"],
                    "Brier Score": results["brier_score"],
                }
            )
            summary[name] = model_summary

        return summary.transpose()


# Example YAML configuration:
"""
models:
  random_forest:
    hyperparameters:
      n_estimators: [100, 500]
      max_depth: [10, 30]
      min_samples_split: [2, 10]
  xgboost:
    hyperparameters:
      n_estimators: [100, 500]
      max_depth: [3, 7]
      learning_rate: [0.01, 0.3]
  catboost:
    hyperparameters:
      iterations: [100, 500]
      depth: [4, 8]
      learning_rate: [0.01, 0.3]
  lightgbm:
    hyperparameters:
      n_estimators: [100, 500]
      num_leaves: [31, 127]
      learning_rate: [0.01, 0.3]

tuning:
  method: optuna  # or 'grid' or 'random'
  parameters:
    n_trials: 100  # for optuna
    n_iter: 100    # for random search
    cv: 5
    n_jobs: -1
"""
