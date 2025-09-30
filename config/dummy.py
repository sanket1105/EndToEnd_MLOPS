import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm

np.seterr(divide="ignore", invalid="ignore")


class ReduceVIF:
    def __init__(self, data: pd.DataFrame, threshold: int = 10, verbose: bool = False):
        self.data = data
        self.threshold = threshold
        self.verbose = verbose

    def calculate_vif_chunks(self, X: pd.DataFrame):
        vif = pd.DataFrame()
        vif["Features"] = X.columns
        for i, col in enumerate(X.columns):
            if X[col].nunique() == 1:
                vif.at[i, "VIF"] = -1
            else:
                vif_value = variance_inflation_factor(X.values, i)
                vif.at[i, "VIF"] = vif_value

        return vif

    def calculate_vif(self, X: pd.DataFrame):
        vif = pd.DataFrame()
        vif["Features"] = X.columns
        pbar = tqdm(
            total=len(X.columns), desc="Calculating VIF", position=0, leave=True
        )
        for i, col in enumerate(X.columns):
            if X[col].nunique() == 1:
                vif.at[i, "VIF"] = -1
            else:
                vif_value = variance_inflation_factor(X.values, i)
                vif.at[i, "VIF"] = vif_value

            pbar.update(1)
        pbar.close()

        vif = vif.sort_values(by="VIF", ascending=False)
        return vif

    def _vif_chunks(self, threshold: int):
        width = self.data.shape[1]
        cols_to_keep = []
        for i in tqdm(
            range(self.data.shape[1] // 50 + 1),
            desc="Processing Chunks",
            position=0,
            leave=True,
            dynamic_ncols=True,
        ):
            start_idx = i * 50
            end_idx = (i + 1) * 50
            df = self.data.iloc[:, start_idx:end_idx]
            df_vif = self.calculate_vif_chunks(df)
            df_vif = df_vif[(df_vif.VIF <= threshold) & (df_vif.VIF > -1)]
            feats_to_keep = list(df_vif.Features)
            cols_to_keep += feats_to_keep

        self.data = self.data.loc[:, cols_to_keep]

        if self.verbose:
            print(f"removed {width - self.data.shape[1]} features")

    def run(self):
        self._vif_chunks(100)
        self._vif_chunks(50)
        self._vif_chunks(25)

        max_vif = 100
        while max_vif > self.threshold:
            vif = self.calculate_vif(self.data)
            # remove infinity ones
            vif = vif[vif.VIF != np.inf]
            vif = vif.reset_index(drop=True)
            # keep all features below threshold or with index of 10+, ensures to aren't removing too many at once
            vif = vif[
                ((vif.VIF <= self.threshold) | (vif.index >= 25)) & (vif.VIF > -1)
            ]
            max_vif = vif.VIF.max()
            self.data = self.data.loc[:, vif.Features]

        return self.data, vif
