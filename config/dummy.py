from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from load_data import load_data


class BuildDataset:
    def __init__(self, data=None):
        self.current_directory = Path(__file__).resolve().parent
        self.data = data

    def read_yaml(self):
        with open(self.current_directory / "config.yaml") as f:
            documents = yaml.full_load(f)

            for item, doc in documents.items():
                setattr(self, item, doc)

        self._prep_general()
        self._prep_environment()
        self._prep_eda()

        self.output_directory = (
            self.current_directory / Path(self.output_folder) / Path(self.subfolders[0])
        )

    def _prep_general(self):
        for item, doc in self.general.items():
            setattr(self, item, doc)
        self.training_date = pd.to_datetime(self.training_date)
        self.performance_date = pd.to_datetime(self.performance_date)

    def _prep_environment(self):
        for item, doc in self.environment.items():
            setattr(self, item, doc)

    def _prep_eda(self):
        for item, doc in self.eda.items():
            setattr(self, item, doc)

    def load_raw_dataset(self):
        if self.verbose:
            print("loading data...")

        if self.data_folder:
            directory = self.current_directory / Path("data")

        df = load_data(
            path=directory / self.dataset["name"],
            file_type=self.dataset["file_type"],
            sheet_name=self.dataset.get("sheet_name"),
            header=self.dataset["header"],
        )

        self.raw = df

        if self.verbose:
            print("data loaded")

    def split_dataset(self):
        # this currently assumes they are split via dataset column
        if self.verbose:
            print("splitting data into training and performance...")

        training_raw = self.raw[self.raw.dataset == "training"]
        performance_raw = self.raw[self.raw.dataset == "performance"]
        self.training_raw = training_raw
        self.performance_raw = performance_raw

        if self.verbose:
            print("data split into training and performance")

    def clean_dataset(self):
        training = self.training_raw.copy(deep=True)
        performance = self.performance_raw.copy(deep=True)
        if self.verbose:
            print(f"original training dataset size {training.shape}")
            width = training.shape[1]

        # fix column names
        training.columns = (
            training.columns.str.strip().str.lower().str.replace(" ", "_")
        )
        cols = training.columns
        performance.columns = cols
        if self.verbose:
            print("column names fixed")

        # remove duplicated columns
        training = training.loc[:, ~training.columns.duplicated()]
        cols = self._save_removed_columns(cols, training.columns, "duplicated")
        if self.verbose:
            print(f"removed {width - training.shape[1]} duplicated features")
            width = training.shape[1]

        # remove empty features
        training = training.dropna(axis=1, how="all")
        cols = self._save_removed_columns(cols, training.columns, "empty")
        if self.verbose:
            print(f"removed {width - training.shape[1]} empty features")
            width = training.shape[1]

        # clean string fields
        training.replace(r"-^\s*$", np.nan, regex=True, inplace=True)
        training.fillna(np.nan, inplace=True)
        training = training.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        performance.replace(r"-^\s*$", np.nan, regex=True, inplace=True)
        performance.fillna(np.nan, inplace=True)
        performance = performance.applymap(
            lambda x: x.strip() if isinstance(x, str) else x
        )

        # remove time fields
        training = training.loc[
            :, [col for col in training.columns if "_time_" not in col]
        ]
        cols = self._save_removed_columns(cols, training.columns, "time")
        if self.verbose:
            print(f"removed {width - training.shape[1]} time features")
            width = training.shape[1]

        # remove missing percentage
        missing_percent = training.isnull().sum() / len(training)
        missing_data_cols = missing_percent[
            missing_percent > self.missing_perc_cuttoff
        ].index
        training.drop(columns=missing_data_cols, inplace=True)
        cols = self._save_removed_columns(cols, training.columns, "high_null")
        if self.verbose:
            print(
                f"removed {width - training.shape[1]} features because > {self.missing_perc_cuttoff * 100}% empty"
            )
            width = training.shape[1]

        #         # remove columns that are exactly equal to each other
        #         original_dtypes = training.dtypes.to_dict()
        #         training = training.T.drop_duplicates(keep='first').T
        #         # reassign dtypes
        #         for col, dtype in original_dtypes.items():
        #             if col not in training.columns:
        #                 continue
        #             training[col] = training[col].astype(dtype)
        #         cols = self._save_removed_columns(cols, training.columns, "exactly_equal")
        #         if self.verbose:
        #             print(f"removed {width - training.shape[1]} exactly equal features")
        #             width = training.shape[1]

        # convert dates to datetime format and convert months since
        date_cols_to_convert = training.select_dtypes(include="object").filter(
            regex="^(lst_|max_)"
        )
        for col in date_cols_to_convert.columns:
            training[col] = training[col].apply(pd.to_datetime, errors="coerce")
            training[col] = (self.training_date - training[col]).astype(
                "timedelta64[M]"
            )
            performance[col] = performance[col].apply(pd.to_datetime, errors="coerce")
            performance[col] = (self.performance_date - performance[col]).astype(
                "timedelta64[M]"
            )
        if self.verbose:
            print("dates converted to time elapsed features")
            width = training.shape[1]

        #         # removes features with < selected variance
        #         var = training.var()
        #         mu = training.mean()
        #         var_to_mu = var / mu
        #         var_mu_cols = var_to_mu[var_to_mu < self.variance_min].index
        #         if "target" in var_mu_cols:
        #             var_mu_cols = var_mu_cols.drop("target")
        #         training.drop(columns=var_mu_cols, inplace=True)
        #         cols = self._save_removed_columns(cols, training.columns, "low_variance")
        #         if self.verbose:
        #             print(f"removed {width - training.shape[1]} features because < {self.variance_min * 100}% variance")
        #             width = training.shape[1]

        # try to convert object columns to numeric if possible else strip spaces
        object_cols = training.select_dtypes(include="object").columns
        for col in object_cols:
            training = self._convert_to_numeric(training, col)

        # set lg_seq_num as index
        training = training.reset_index().rename(columns={"index": "lg_seq_index"})
        performance = performance.reset_index().rename(
            columns={"index": "lg_seq_index"}
        )

        # store features
        features = pd.DataFrame(data=training.columns, columns=["features"])
        features["dtypes"] = training.dtypes.values
        self.features = features

        # set performance to only use training columns
        performance = performance.loc[:, training.columns]

        if self.verbose:
            print(f"removed a total of {self.raw.shape[1] - width} features")
            print(f"final training dataset size {training.shape}")
        self.training = training
        self.performance = performance

    def _save_removed_columns(self, old, new, name):
        removed_cols = list(set(old).symmetric_difference(set(new)))
        with open(f"{self.output_directory}/features_{name}.txt", "w") as f:
            for item in removed_cols:
                f.write("%s\n" % item)

        return new

    def _convert_to_numeric(self, df, col):
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            df[col] = df[col].str.strip()

        return df

    def run(self):
        self.read_yaml()
        if self.data is None:
            self.load_raw_dataset()
        else:
            if self.verbose:
                print("data preloaded")
            self.raw = self.data
        self.split_dataset()
        self.clean_dataset()

        self.features.to_csv(
            f"{self.output_directory}/dataset_features.csv", index=False
        )
        self.training.to_csv(
            f"{self.output_directory}/cleaned_training_data.csv", index=False
        )
        self.performance.to_csv(
            f"{self.output_directory}/cleaned_performance_data.csv", index=False
        )

        return self.training, self.performance
