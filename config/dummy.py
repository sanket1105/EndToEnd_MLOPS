import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


class BinData:
    def __init__(
        self,
        X_train,
        X_valid,
        binning_cols,
        non_binning_cols,
        n_bins,
        bin_encoding,
        bin_strategy,
        output_path,
    ):
        self.X_train = X_train
        self.X_valid = X_valid
        self.binning_cols = binning_cols
        self.non_binning_cols = non_binning_cols
        self.n_bins = n_bins
        self.bin_encoding = bin_encoding
        self.bin_strategy = bin_strategy
        self.output_path = output_path

    def split_data(self):
        self.X_train_bin = self.X_train.loc[:, self.binning_cols]
        self.X_train_non_bin = self.X_train.loc[:, self.non_binning_cols]
        self.X_valid_bin = self.X_valid.loc[:, self.binning_cols]
        self.X_valid_non_bin = self.X_valid.loc[:, self.non_binning_cols]

    def bin_data(self):
        est = KBinsDiscretizer(
            n_bins=self.n_bins, encode=self.bin_encoding, strategy=self.bin_strategy
        )
        X_train_kbins = pd.DataFrame.sparse.from_spmatrix(
            est.fit_transform(self.X_train_bin)
        )
        X_valid_kbins = pd.DataFrame.sparse.from_spmatrix(
            est.transform(self.X_valid_bin)
        )

        self.est = est
        self.X_train_kbins = X_train_kbins
        self.X_valid_kbins = X_valid_kbins

    def update_columns(self):
        new_cols = []
        for j in range(len(self.binning_cols)):
            for k in range(len(self.est.bin_edges_[j]) - 1):
                base = self.binning_cols[j]
                lower_val = round(self.est.bin_edges_[j][k], 4)
                upper_val = round(self.est.bin_edges_[j][k + 1], 4)
                col_name = f"{base}_{lower_val}_{upper_val}"
                new_cols.append(col_name)
        self.X_train_kbins.columns = new_cols
        self.X_valid_kbins.columns = new_cols

    def add_non_transformed_data(self):
        self.X_train_kbins = pd.concat(
            [self.X_train_non_bin, self.X_train_kbins], axis=1
        )
        self.X_valid_kbins = pd.concat(
            [self.X_valid_non_bin, self.X_valid_kbins], axis=1
        )

    def run(self):
        self.split_data()
        self.bin_data()
        self.update_columns()
        self.add_non_transformed_data()

        return self.X_train_kbins, self.X_valid_kbins, self.est
