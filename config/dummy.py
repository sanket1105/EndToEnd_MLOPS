import json
import pickle
from typing import Tuple, Union

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from colorama import Fore
from matplotlib.figure import Figure
from pyspark.sql import DataFrame
from scipy import stats
from scipy.stats import anderson, kstest, shapiro
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder


def remove_features(
    df: pd.DataFrame, excluded_feats: list, verbose: bool = False
) -> pd.DataFrame:
    # TODO: docstring
    n_cols = df.shape[1]
    excluded_feats = [
        feat.replace('"', "").replace("\\", "").replace(" ", "_").lower()
        for feat in excluded_feats
    ]
    excluded_feats = [feat for feat in excluded_feats if feat in df.columns]

    df.drop(columns=excluded_feats, inplace=True)
    if verbose:
        print(f"{n_cols - df.shape[1]} cols removed")

    return df


def pyspark_remove_features(
    df: DataFrame, excluded_feats: list, verbose: bool = False
) -> DataFrame:
    # TODO: docstring
    n_cols = len(df.columns)
    excluded_feats = [
        feat.replace('"', "").replace("\\", "").replace(" ", "_").lower()
        for feat in excluded_feats
    ]
    excluded_feats = [feat for feat in excluded_feats if feat in df.columns]

    df = df.drop(*excluded_feats)
    if verbose:
        print(f"{n_cols - len(df.columns)} cols removed")

    return df


def category_nulls(
    df: pd.DataFrame, cols: list, method: str, verbose: bool = False
) -> pd.DataFrame:
    # TODO: add assert statement
    # TODO: docstring
    df = df.copy(deep=True)
    if verbose:
        print(f"handling category nulls with {method} method")
        length = df.shape[0]
    if method == "unknown":
        for c in cols:
            df[c] = df[c].fillna("unknown")
    elif method == "mode":
        for c in cols:
            mode = df[c].mode()[0]
            df[c] = df[c].fillna(mode)
    elif method == "drop":
        df = df.dropna(subset=cols)
    if verbose:
        if method == "drop":
            print(f"removed {length - df.shape[0]} category null rows")
        else:
            print("category nulls imputed")
    return df


def float_nulls(
    df: pd.DataFrame,
    cols: list,
    method: str,
    replacement_value: int = -1,
    verbose: bool = False,
) -> pd.DataFrame:
    # TODO: add assert statement
    # TODO: docstring
    df = df.copy(deep=True)
    if verbose:
        print(f"replacing float nulls with {method} method")
        length = df.shape[0]
    if method == "replace":
        for c in cols:
            df[c] = df[c].fillna(replacement_value)
    elif method == "mode":
        for c in cols:
            mode = df[c].mode()[0]
            df[c] = df[c].fillna(mode)
    elif method == "drop":
        df = df.dropna(subset=cols)
    if verbose:
        if method == "drop":
            print(f"removed {length - df.shape[0]} null rows")
        else:
            print("float nulls imputed")
    return df


def replace_nulls(
    df: pd.DataFrame, method: str, verbose: bool = False
) -> Tuple[pd.DataFrame, dict]:
    # TODO: add assert statement
    # TODO: docstring
    df = df.copy(deep=True)
    replacement_dict = {}
    if verbose:
        print(f"imputing nulls with {method} method")
    for col in df.columns:
        num_na = df[col].isna().sum()
        if num_na == 0:
            continue

        if method == "mean":
            fill_value = df[col].mean()
        elif method == "median":
            fill_value = df[col].median()
        elif method == "mode":
            fill_value = df[col].mode()[0]

        df[col] = df[col].fillna(fill_value)
        replacement_dict[col] = fill_value

    if verbose:
        print("nulls imputed")

    return df, replacement_dict


def data_countplot(
    df: pd.DataFrame,
    col: str,
    xticks_rotation: int = 0,
    title: str = None,
    figsize: Tuple[int, int] = (8, 6),
    savefig: bool = True,
    path: str = None,
) -> Figure:
    # TODO: docstring
    sns.set_style("darkgrid")
    fig = plt.figure(figsize=figsize)

    ax = sns.countplot(data=df, x=col, label="Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xticks_rotation, ha="right")
    ax.set_title(title)
    ax.tick_params(
        top=False,
        bottom=False,
        left=False,
        right=False,
        labelleft=True,
        labelbottom=True,
    )
    plt.tight_layout()
    if savefig:
        plt.savefig(path)
    return fig


def encode_data(
    df: pd.DataFrame, col: str, method, path
) -> Tuple[pd.DataFrame, Union[LabelEncoder, OneHotEncoder, OrdinalEncoder]]:
    # TODO: add assert statement
    # TODO: docstring
    if method == "label":
        encoder = LabelEncoder()
    elif method == "one_hot":
        encoder = OneHotEncoder()
    elif method == "orginal":
        encoder = OrdinalEncoder()
    encoded_values = encoder.fit_transform(df[col])

    # create new column(s) with the encoded values
    if method == "one_hot":
        # For OneHotEncoder, you may need to handle sparse matrices or convert them to dense
        encoded_columns = pd.DataFrame(
            encoded_values, columns=encoder.get_feature_names([col])
        )
        df_encoded = pd.concat([df, encoded_columns], axis=1)
    else:
        df_encoded = df.copy(deep=True)
        df_encoded[col] = encoded_values
    with open(f"{path}/encoder_{col}.pkl", "wb") as f:
        pickle.dump(encoder, f)
    return df_encoded, encoder


def get_data_sample(
    df: pd.DataFrame,
    path: str,
    target: str = "target",
    selected_response_ratio: float = 0.1,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, dict, Figure]:
    # TODO: docstring
    counts = df[target].value_counts(normalize=False)
    non_response = int(counts[0])
    response = int(counts[1])
    non_response_ratio = non_response / (non_response + response)
    response_ratio = 1 - non_response_ratio
    desired_non_response = int(response / selected_response_ratio - response)

    if verbose:
        print(f"Non-Respondents: \t{non_response} \t({non_response_ratio*100:.2f}%)")
        print(f"Respondents: \t\t{response} \t({response_ratio*100:.2f}%)")
        print()

        print(
            f"Desired response percent of population: \t{selected_response_ratio*100:.0f}%"
        )
        print(f"Desired sample non-respondents: \t\t{desired_non_response}")
        print(f"Desired sample respondents: \t\t\t{response}")
        print()

        print(
            f"Desired non-respondents:non-respondents ratio: {desired_non_response / non_response:.4f}"
        )
        print(
            f"Non-Respondents:desired non-respondents ratio: {non_response / desired_non_response:.4f}"
        )
    sampling_dict = {
        "pop_non_response": non_response,
        "pop_response": response,
        "pop_non_response_ratio": round(non_response_ratio, 4),
        "pop_response_ratio": round(response_ratio, 4),
        "sample_non_response": desired_non_response,
        "sample_response": response,
    }

    # save sampling dict
    with open(f"{path}/sampling_dict.json", "w") as outfile:
        json.dump(sampling_dict, outfile)
    # make plot
    fig = data_countplot(
        df=df,
        col=target,
        path=f"{path}/response_countplot.png",
        title="Population Response Distribution",
        figsize=(8, 6),
        savefig=True,
    )

    # subset data
    df_sampled = pd.concat(
        [
            df[df[target] == 1],
            df[df[target] == 0].sample(
                n=desired_non_response, replace=False, random_state=0
            ),
        ],
        axis=0,
    )

    return df_sampled, sampling_dict, fig


def transformation_significance(
    X: pd.DataFrame,
    trans_score_df: pd.DataFrame,
    sig_level: float = 0.05,
    verbose: bool = False,
) -> pd.DataFrame:
    # TODO: docstring
    if verbose:
        print("running anova test on transformations")
    trans_score_df = trans_score_df.copy(deep=True)
    trans_score_df["transformation_override"] = "nominal"

    overridden_transformations = []
    for idx, row in trans_score_df.iterrows():
        if row.transformation == "nominal":
            continue
        tmp_anova = X.loc[:, [idx + "_nominal", idx + "_" + row.transformation]]

        tmp_anova.dropna(inplace=True)

        anova = stats.f_oneway(tmp_anova.iloc[:, 0], tmp_anova.iloc[:, 1])
        if anova.pvalue >= sig_level:
            overridden_transformations.append(idx)
        else:
            trans_score_df.loc[idx, "transformation_override"] = row.transformation

    if verbose:
        print(
            f"{len(overridden_transformations)} features changed to nominal for lack of statistical significance"
        )
        print(overridden_transformations)

    return trans_score_df


def low_variance(df_train: DataFrame, df_test: DataFrame, threshold: float):
    cols_to_remove = []

    for i in df_train.columns:
        if df_train[i].var() < 0.01:
            cols_to_remove.append(i)

    df_train = df_train.drop(cols_to_remove, axis=1)
    df_test = df_test.drop(cols_to_remove, axis=1)

    return df_train, df_test, cols_to_remove


def test_normality(df: pd.DataFrame, numeric_cols: list):
    """
    Apply Shapiro-Wilk, Kolmogorov-Smirnov, and Anderson-Darling tests
    to check normality on selected numeric columns in a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        numeric_cols (list): List of column names to test for normality.

    Returns:
        dict: A dictionary with results for each test ('shapiro', 'kstest', 'anderson'),
              listing the columns that passed the normality test.
    """

    # Dictionary to store results for each test
    normality_results = {"shapiro": [], "kstest": [], "anderson": []}

    # Utility function for Shapiro-Wilk test
    def apply_shapiro(series):
        stat, p_value = shapiro(series)
        return p_value > 0.05

    # Utility function for Kolmogorov-Smirnov test
    def apply_kstest(series):
        stat, p_value = kstest(series, "norm")
        return p_value > 0.05

    # Utility function for Anderson-Darling test
    def apply_anderson(series):
        result = anderson(series)
        return all(
            result.statistic < result.critical_values[i]
            for i in range(len(result.critical_values))
        )

    # General function to apply a given test to a column
    def apply_test(series, test_func, test_name):
        if test_func(series):
            normality_results[test_name].append(series.name)

    # Loop over the selected numeric columns and apply each test
    for column in numeric_cols:
        if column in df.columns:  # Check if the column exists in the DataFrame
            apply_test(df[column], apply_shapiro, "shapiro")
            apply_test(df[column], apply_kstest, "kstest")
            apply_test(df[column], apply_anderson, "anderson")

    return normality_results


def remove_correlated(
    X_train: pd.DataFrame, threshold: float
) -> (pd.DataFrame, list, list):
    """
    First remove the columns with just one unique value in it and then removes highly correlated features from the DataFrame.

    Args:
        X_train (DataFrame): Input DataFrame (training set).
        threshold (float): The correlation threshold above which columns will be dropped.

    Returns:
        pd.DataFrame: DataFrame with highly correlated features removed.
        list: List of columns that were dropped because of one unique feature
        list: List of columns that were dropped because of high correlation

    """

    # Step 1: Remove columns with only one unique value
    one_unique_feature = [col for col in X_train.columns if X_train[col].nunique() < 2]

    X_train.drop(columns=one_unique_feature, inplace=True)
    print(
        f"Dropped {len(one_unique_feature)} columns with one unique value: {one_unique_feature}"
    )

    corr = X_train.corr().abs()
    ## selecting the upper triangle
    upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    correlated_pairs = []
    for column in upper_tri.columns:
        correlated_features = upper_tri[column][
            upper_tri[column] > threshold
        ].index.tolist()
        if correlated_features:
            for feature in correlated_features:
                correlated_pairs.append((column, feature, upper_tri[column][feature]))

    # Drop the features with high correlation
    to_drop = [
        column for column in upper_tri.columns if any(upper_tri[column] > threshold)
    ]
    X_train_cleaned = X_train.drop(columns=to_drop)

    # Display the dropped columns and their correlations
    print("Dropped Columns and their correlations:")
    for col1, col2, corr_value in correlated_pairs:
        print(
            f"'{col1}' is correlated with '{col2}' with a correlation of {corr_value:.2f}"
        )

    return X_train_cleaned, one_unique_feature, to_drop


def ks_lift_metrics(
    data, target, prob, prior_prob, pop_1, pop_0, n_buckets=10, summary_cols=[]
):
    """
    Calculate KS & Lift metrics at population level.

    Parameters:
    ------------
    data : pandas DataFrame
        The dataset containing target and probability columns.
    target : str
        The name of the column containing the actual target values (binary).
    prob : str
        The name of the column containing predicted probabilities.
    prior_prob : float
        The prior probability value to be adjusted.
    pop_1 : int
        The size of population with target=1 for adjusted metrics.
    pop_0 : int
        The size of population with target=0 for adjusted metrics.
    n_buckets : int, optional (default=10)
        Number of quantiles/buckets to split data into.
    summary_cols : list, optional (default=[])
        List of additional summary columns to include in the output.

    Returns:
    --------
    kstable : pandas DataFrame
        DataFrame with calculated KS, Lift and other related metrics.
    """

    # Sort by probability and create target0 column
    data = data.sort_values(by=prob, ascending=False).reset_index(drop=True)
    data["target0"] = 1 - data[target]

    # Bucket the data into quantiles
    data["bucket"] = pd.qcut(data[prob].rank(method="first"), n_buckets)
    grouped = data.groupby("bucket", as_index=False)

    # Initialize the KS table
    kstable = pd.DataFrame()
    kstable["min_prob"] = grouped.min()[prob]
    kstable["max_prob"] = grouped.max()[prob]
    kstable["accounts"] = grouped.count()[target]
    kstable["events"] = grouped.sum()[target]
    kstable["nonevents"] = grouped.sum()["target0"]

    # Add any summary columns
    for col in summary_cols:
        kstable[col] = grouped.mean()[col]

    # Sort by max_prob and min_prob
    kstable = kstable.sort_values(
        by=["max_prob", "min_prob"], ascending=[False, False]
    ).reset_index(drop=True)

    # Event rates, KS statistic, Lift, and other metrics
    kstable["event_rate"] = (kstable.events / data[target].sum()).apply(
        "{0:.2%}".format
    )
    kstable["nonevent_rate"] = (kstable.nonevents / data["target0"].sum()).apply(
        "{0:.2%}".format
    )
    kstable["cum_event_rate"] = (kstable.events / data[target].sum()).cumsum()
    kstable["cum_nonevent_rate"] = (kstable.nonevents / data["target0"].sum()).cumsum()
    kstable["KS"] = (
        np.abs(np.round(kstable["cum_event_rate"] - kstable["cum_nonevent_rate"], 3))
        * 100
    )
    kstable["Interval_RR"] = np.round(kstable["events"] / kstable["accounts"], 3) * 100
    kstable["Lift"] = np.round(
        kstable["Interval_RR"]
        / (np.round(kstable["events"].sum() / kstable["accounts"].sum(), 3) * 100),
        2,
    )

    # Bias probability and adjusted metrics
    bias_prob = kstable["events"].sum() / kstable["accounts"].sum()
    prior_prob = prior_prob / 100

    kstable["IRR"] = kstable["Interval_RR"] / 100
    kstable["Adj_Interval_RR"] = (
        (kstable["IRR"] * (prior_prob / bias_prob))
        / (
            ((1 - kstable["IRR"]) * ((1 - prior_prob) / (1 - bias_prob)))
            + (kstable["IRR"] * (prior_prob / bias_prob))
        )
    ) * 100
    kstable["Adj_Lift"] = kstable["Adj_Interval_RR"] / (prior_prob * 100)

    # Adjusted events and nonevents
    kstable["Adj_events"] = (kstable["Adj_Interval_RR"] / 100) * kstable["accounts"]
    kstable["Adj_nonevents"] = (1 - kstable["Adj_Interval_RR"] / 100) * kstable[
        "accounts"
    ]

    # Weighting based on population size
    wt_pop_1 = pop_1 / kstable["Adj_events"].sum()
    wt_pop_0 = pop_0 / kstable["Adj_nonevents"].sum()
    kstable["Adj_events_pop"] = np.round(wt_pop_1 * kstable["Adj_events"], 0)
    kstable["Adj_nonevents_pop"] = np.round(wt_pop_0 * kstable["Adj_nonevents"], 0)
    kstable["Adj_accounts_pop"] = np.round(
        kstable["Adj_events_pop"] + kstable["Adj_nonevents_pop"], 0
    )

    # Population rates
    kstable["events_rate_pop"] = np.round((kstable["Adj_events_pop"] / pop_1) * 100, 1)
    kstable["nonevents_rate_pop"] = np.round(
        (kstable["Adj_nonevents_pop"] / pop_0) * 100, 1
    )
    kstable["cum_events_rate_pop"] = np.round(
        (kstable["Adj_events_pop"] / pop_1).cumsum() * 100, 1
    )
    kstable["cum_nonevents_rate_pop"] = np.round(
        (kstable["Adj_nonevents_pop"] / pop_0).cumsum() * 100, 1
    )
    kstable["Interval_RR_pop"] = (
        np.round(kstable["Adj_events_pop"] / kstable["Adj_accounts_pop"], 3) * 100
    )
    kstable["KS_pop"] = np.abs(
        np.round(kstable["cum_events_rate_pop"] - kstable["cum_nonevents_rate_pop"], 3)
    )

    # Decumulative events and nonevents
    kstable["decum_nonevents"] = (
        100 - kstable["cum_nonevents_rate_pop"].shift(1)
    ).fillna(100)
    kstable["decum_events"] = (100 - kstable["cum_events_rate_pop"].shift(1)).fillna(
        100
    )
    kstable["cum_RR_pop"] = ""

    # Select the relevant output columns
    out_cols = [
        "min_prob",
        "max_prob",
        "Adj_accounts_pop",
        "Adj_nonevents_pop",
        "nonevents_rate_pop",
        "cum_nonevents_rate_pop",
        "decum_nonevents",
        "Adj_events_pop",
        "events_rate_pop",
        "cum_events_rate_pop",
        "decum_events",
        "Interval_RR_pop",
        "cum_RR_pop",
        "KS_pop",
    ] + summary_cols

    kstable = kstable[out_cols]

    # Set index and format output
    kstable.index = range(1, n_buckets + 1)
    kstable.index.rename("Group", inplace=True)
    pd.set_option("display.max_columns", 12)

    # Display KS statistic
    print(
        Fore.RED
        + "KS is "
        + str(np.round(max(kstable["KS_pop"]), 1))
        + "%"
        + " at decile "
        + str((kstable.index[kstable["KS_pop"] == max(kstable["KS_pop"])][0]))
    )

    return kstable


def log_mlflow_run(
    model, X_train, X_test, y_train, y_test, model_name="model", params=None
):
    """
    Function to log model training and evaluation to MLFlow.

    Parameters:
        model (sklearn model): The model to train and log.
        X_train, X_test, y_train, y_test (arrays): Training and test datasets.
        model_name (str): Name to log the model under.
        params (dict): Parameters to log in MLFlow.
    """
    with mlflow.start_run(run_name=model_name) as run:
        # Fit the model
        model.fit(X_train, y_train)

        # Log parameters if provided
        if params:
            mlflow.log_params(params)

        # Run evaluation and log metrics
        baseline_scores, _ = EvaluateModel(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            model_name=model_name,
        ).run()

        # Log metrics
        mlflow.log_metrics(baseline_scores)

        # Log the model itself
        mlflow.sklearn.log_model(model, model_name)

    return baseline_scores
