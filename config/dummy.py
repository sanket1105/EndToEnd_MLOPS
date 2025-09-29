import numpy as np
import pandas as pd

# from scipy.stats import zscore
from tqdm import tqdm


def _outlier_z_score(df, col, **kwargs):
    # calculate z-scores
    # df["z_score"] = zscore(df[col])

    # calculate mean and variance
    mu = df[col].mean()
    # var = df[col].var()
    std = df[col].std()

    # calculate thresholds
    neg_z = -3 * std + mu
    pos_z = 3 * std + mu

    # # apply thresholding
    # df[col] = np.where(df.z_score < -3, neg_z, np.where(df.z_score > 3, pos_z, df[col]))

    # # drop z-scores
    # df.drop(columns=["z_score"], inplace=True)

    # return df, neg_z, pos_z
    return neg_z, pos_z


def _outlier_percentile(df, col, lower_percentile=1, upper_percentile=99, **kwargs):
    # calculate bounds
    lower_bound = np.percentile(df[col], lower_percentile)
    upper_bound = np.percentile(df[col], upper_percentile)

    # # apply thresholding
    # df[col] = np.where(
    #     df[col] < lower_bound,
    #     lower_bound,
    #     np.where(df[col] > upper_bound, upper_bound, df[col]),
    # )

    # return df, lower_bound, upper_bound
    return lower_bound, upper_bound


def _outlier_iqr(df, col, iqr_threshold=1.5, **kwargs):
    # calculate iqr
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    # calculate outlier threshold
    lower_bound = q1 - iqr_threshold * iqr
    upper_bound = q3 + iqr_threshold * iqr

    # # apply thresholding
    # df[col] = np.where(
    #     df[col] < lower_bound,
    #     lower_bound,
    #     np.where(df[col] > upper_bound, upper_bound, df[col]),
    # )

    # return df, lower_bound, upper_bound
    return lower_bound, upper_bound


def handle_outliers(
    data: pd.DataFrame,
    technique: str,
    cols: str or list = "All",
    verbose: bool = False,
    **kwargs
) -> pd.DataFrame:
    df = data.copy(deep=True)

    if verbose:
        print("removing outliers...")

    if cols == "All":
        cols = df.columns

    # remove non-numeric columns
    cols = df.loc[:, cols].select_dtypes(include=[np.number]).columns

    # remove columns with with less than 3 unique values
    cols = [c for c in cols if len(df[c].dropna().unique()) > 2]

    outlier_bounds = {}
    if technique == "z-score":
        for c in tqdm(cols):
            # df, neg_z, pos_z = _outlier_z_score(df, c, **kwargs)
            neg_z, pos_z = _outlier_z_score(df, c, **kwargs)
            outlier_bounds[c] = {
                "lower_bound": neg_z,
                "upper_bound": pos_z,
            }
    elif technique == "percentile":
        for c in tqdm(cols):
            # df, lower_bound, upper_bound = _outlier_percentile(df, c, **kwargs)
            lower_bound, upper_bound = _outlier_percentile(df, c, **kwargs)
            outlier_bounds[c] = {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }
    elif technique == "iqr":
        for c in tqdm(cols):
            # df, lower_bound, upper_bound = _outlier_iqr(df, c, **kwargs)
            lower_bound, upper_bound = _outlier_iqr(df, c, **kwargs)
            outlier_bounds[c] = {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }

    if verbose:
        print("outliers removed")

    # return df, outlier_bounds
    return outlier_bounds


if __name__ == "__main__":
    raw_data = {
        "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 1000],
        "income": [
            50000,
            60000,
            75000,
            80000,
            90000,
            120000,
            125000,
            150000,
            200000,
            250000,
            300000,
        ],
        "fruit": [
            "apple",
            "banana",
            "orange",
            "grape",
            "cherry",
            "pear",
            "kiwi",
            "mango",
            "pineapple",
            "strawberry",
            "watermelon",
        ],
    }
    df = pd.DataFrame(raw_data)
    mydict = handle_outliers(df, "z-score", cols=["age", "income"], verbose=True)
    print(mydict)
