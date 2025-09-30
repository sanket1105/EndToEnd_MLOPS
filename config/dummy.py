import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


class TransformationEvaluator:
    def __init__(self, X: pd.DataFrame, y: pd.Series, scoring_method: str) -> float:
        self.X = X.copy(deep=True)
        self.y = y.copy(deep=True)
        self.scoring_method = scoring_method

        assert scoring_method in (
            "accuracy",
            "precision",
            "recall",
            "auc",
            "f1",
            "log_loss",
            "r2",
            "matthews",
        ), "scoring method must be one of accuracy, precision, recall, auc, f1, log loss, r2, matthews"

    def train(self):
        self.X = sm.add_constant(self.X)
        model = sm.Logit(self.y, self.X).fit()
        self.r2 = model.prsquared

        self.model = model

    def predict(self):
        pos_prob = self.model.predict(self.X)
        neg_prob = 1 - pos_prob
        preds = round(pos_prob)
        # probs = np.column_stack((neg_prob, pos_prob))
        probs = pos_prob
        self.preds = preds
        self.probs = probs

    def score(self):
        y_true = self.y
        y_pred = self.preds
        y_prob = self.probs
        if self.scoring_method == "accuracy":
            return accuracy_score(y_true, y_pred)
        elif self.scoring_method == "precision":
            return precision_score(y_true, y_pred)
        elif self.scoring_method == "recall":
            return recall_score(y_true, y_pred)
        elif self.scoring_method == "f1":
            return f1_score(y_true, y_pred)
        elif self.scoring_method == "auc":
            return roc_auc_score(y_true, y_prob)
        elif self.scoring_method == "log_loss":
            return log_loss(y_true, y_prob)
        elif self.scoring_method == "r2":
            return self.r2
        elif self.scoring_method == "matthews":
            return matthews_corrcoef(y_true, y_pred)

    def run(self):
        self.train()
        self.predict()
        model_score = self.score()

        return model_score


if __name__ == "__main__":
    data = pd.DataFrame(
        {"grade_nominal": [75, 90, 80, 83, 100], "passing": [0, 1, 1, 0, 1]}
    )
    X = data.iloc[:, :1]
    y = data.iloc[:, -1]
    TransformationEvaluator(X, y, "accuracy").run()
    TransformationEvaluator(X, y, "precision").run()
    TransformationEvaluator(X, y, "recall").run()
    TransformationEvaluator(X, y, "f1").run()
    TransformationEvaluator(X, y, "log_loss").run()
    TransformationEvaluator(X, y, "r2").run()
    TransformationEvaluator(X, y, "matthews").run()
