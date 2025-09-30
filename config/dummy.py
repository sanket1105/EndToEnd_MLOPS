import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)


class DataTransformer:
    def __init__(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        cols: list,
        transformations: list,
    ):
        """
        Initialize the DataTransformer.

        :param train: Training DataFrame.
        :param valid: Validation DataFrame.
        :param cols: List of columns to transform.
        :param transformations: List of transformation names.
        """
        self.train = train
        self.valid = valid
        self.cols = cols
        self.transformations = transformations
        self.transformations_dict = {
            "log": FunctionTransformer(np.log, validate=True),
            "sqrt": FunctionTransformer(np.sqrt, validate=True),
            "reciprocal": FunctionTransformer(np.reciprocal, validate=True),
            "exponential": FunctionTransformer(lambda x: x**2, validate=True),
            "yeo_johnson": PowerTransformer(
                method="yeo-johnson", standardize=False, copy=False
            ),
        }

        self.transformer_objects = {}
        self.transformer_details = {}
        self.small_constant = 1e-8  # Defined constant for log and reciprocal

    def fill_nans_with_mean(self, df):
        """Fill NaNs with column means."""
        return df.fillna(df.mean())

    def apply_transformation(self, transformer):
        func = self.transformations_dict[transformer]

        train_filled = self.fill_nans_with_mean(self.train[self.cols].copy(deep=True))
        valid_filled = self.fill_nans_with_mean(self.valid[self.cols].copy(deep=True))

        if transformer in ["log", "reciprocal"]:
            train_filled += self.small_constant
            valid_filled += self.small_constant

        # Fit the PowerTransformer for Yeo-Johnson
        if transformer == "yeo_johnson":
            func.fit(train_filled)
            self.transformer_objects[transformer] = func
            self.transformer_details[transformer] = dict(zip(self.cols, func.lambdas_))

        df_train = pd.DataFrame(func.transform(train_filled), columns=self.cols)
        df_valid = pd.DataFrame(func.transform(valid_filled), columns=self.cols)

        # Replace NaNs that came from log and sqrt transformations
        if transformer in ["log", "sqrt"]:
            df_train.fillna(0, inplace=True)
            df_valid.fillna(0, inplace=True)

        self.train = pd.concat(
            [self.train, df_train.add_suffix(f"_{transformer}")], axis=1
        )
        self.valid = pd.concat(
            [self.valid, df_valid.add_suffix(f"_{transformer}")], axis=1
        )

    def apply_scaler(self, transformer):
        scaler = {
            "min_max": MinMaxScaler(),
            "standard": StandardScaler(),
            "robust": RobustScaler(),
        }[transformer.replace("_scaler", "")]

        df_train = pd.DataFrame(
            scaler.fit_transform(self.train[self.cols]), columns=self.cols
        ).add_suffix(f"_{transformer}")
        df_valid = pd.DataFrame(
            scaler.transform(self.valid[self.cols]), columns=self.cols
        ).add_suffix(f"_{transformer}")

        self.train = pd.concat([self.train, df_train], axis=1)
        self.valid = pd.concat([self.valid, df_valid], axis=1)

        self.transformer_details[transformer] = {
            col_name: (
                {"min": scaler.data_min_, "max": scaler.data_max_}
                if transformer == "min_max_scaler"
                else (
                    {"mu": scaler.mean_, "std": scaler.scale_}
                    if transformer == "standard_scaler"
                    else {"median": scaler.center_, "iqr": scaler.scale_}
                )
            )
            for col_name in self.cols
        }
        self.transformer_objects[transformer] = scaler

    def run(self):
        for t in self.transformations:
            if "_scaler" not in t:
                self.apply_transformation(t)
            else:
                self.apply_scaler(t)

        nominal_cols = [c + "_nominal" for c in self.cols]
        self.train.rename(columns=dict(zip(self.cols, nominal_cols)), inplace=True)
        self.valid.rename(columns=dict(zip(self.cols, nominal_cols)), inplace=True)

        return (
            self.train,
            self.valid,
            self.transformer_objects,
            self.transformer_details,
        )


if __name__ == "__main__":
    train = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Carol"],
            "age": [25, 30, 35],
            "grade": [75, 90, 80],
            "is_senior": [1, 0, 1],
        }
    )
    valid = pd.DataFrame(
        {
            "name": ["Ian", "Mari"],
            "age": [35, 0],
            "grade": [np.nan, 95],
            "is_senior": [1, 0],
        }
    )
    transformation_cols = ["age", "grade"]
    transformations = [
        "log",
        "sqrt",
        "exponential",
        "reciprocal",
        "yeo_johnson",
        "min_max_scaler",
        "standard_scaler",
        "robust_scaler",
    ]

    train_scaled, valid_scaled, trans_objects, trans_details = DataTransformer(
        train=train,
        valid=valid,
        cols=transformation_cols,
        transformations=transformations,
    ).run()

    print(train_scaled)
    print(valid_scaled)
    print(trans_objects)
    print(trans_details)
