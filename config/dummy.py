from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class EvaluateModel:
    def __init__(
        self,
        model,
        X_train: Union[np.array, pd.DataFrame],
        X_test: Union[np.array, pd.DataFrame],
        y_train: pd.Series,
        y_test: pd.Series,
        model_name: str,
        model_type: str,
    ):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_name = model_name
        self.model_type = model_type

    def predict_outcomes(self, X):
        if self.model_name == "majority":
            preds, probs = self._generate_majority_class_preds(y=X)
        else:
            preds = self.model.predict(X)
            probs = self.model.predict_proba(X)

        return preds, probs

    def _generate_majority_class_preds(self, y):
        majority_class = self.y_train.mode()[0]
        base_array = np.zeros(self.y_train.max() + 1)
        base_array[majority_class] = 1

        preds = pd.Series([majority_class] * y.shape[0])
        probs = np.tile(base_array, (y.shape[0], 1))

        return preds, probs

    def score_preds(self, y_true, y_pred, y_prob, dataset):
        mydict = {
            "model": self.model_name,
            "model_type": self.model_type,
            "dataset": dataset,
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "log_loss": log_loss(y_true, y_prob),
            "brier": brier_score_loss(y_true, y_prob[:, 1]),
            "roc_auc": roc_auc_score(y_true, y_prob[:, 1]),
        }

        return mydict

    def plot_roc_curve(self, y_true, y_prob, dataset):
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))

        ns_probs = [0 for _ in range(len(y_true))]
        lr_probs = y_prob[:, 1]
        ns_auc = roc_auc_score(y_true, ns_probs)
        lr_auc = roc_auc_score(y_true, lr_probs)

        # summarize scores
        print("No Skill: ROC AUC=%.3f" % (ns_auc))
        print(f"{self.model_name}: ROC AUC=%.3f" % (lr_auc))
        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(y_true, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(y_true, lr_probs)

        # plot the roc curve for the model
        ax[0].plot(ns_fpr, ns_tpr, linestyle="--", label="No Skill")
        ax[0].plot(lr_fpr, lr_tpr, marker=".", label=f"{self.model_name}")
        ax[0].set_xlabel("False Positive Rate")
        ax[0].set_ylabel("True Positive Rate")
        ax[0].set_title("ROC Curve")
        ax[0].legend()

        # Additional subplot (you can modify this part as needed)
        ax[1].hist(lr_probs, bins=50, label=f"{self.model_name} Probabilities")
        ax[1].set_xlim(0, 1)
        ax[1].set_xlabel("Probability")
        ax[1].set_ylabel("Frequency")
        ax[1].set_title(f"Distribution of Probabilities")
        ax[1].legend()

        plt.suptitle(
            f"{self.model_type.title()} {self.model_name.title()} Model for {dataset.title()}".strip(),
            fontsize=16,
        )

        plt.tight_layout()  # Ensures proper spacing between subplots
        plt.show()

    def run(self):
        y_pred_train, y_prob_train = self.predict_outcomes(self.X_train)
        y_pred, y_prob = self.predict_outcomes(self.X_test)
        train_scores = self.score_preds(
            self.y_train, y_pred_train, y_prob_train, "train"
        )
        test_scores = self.score_preds(self.y_test, y_pred, y_prob, "test")
        scores = pd.DataFrame([train_scores, test_scores])

        self.plot_roc_curve(self.y_train, y_prob_train, "training")
        self.plot_roc_curve(self.y_test, y_prob, "testing")

        return scores, (y_prob_train, y_prob)
