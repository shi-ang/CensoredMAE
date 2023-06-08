import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from pycox.datasets import metabric
from typing import Tuple

from util_survival import compare_km_curves, KaplanMeier


def make_semi_synth_mimiciv(
        data='icu',
        censor_dist: str = 'Uniform'
) -> pd.DataFrame:
    # Using the SQL query from Survival MDN paper
    if data == 'icu':
        df_full = pd.read_csv('data/MIMIC/MIMIC_IV_all_cause_failure.csv')
        skip_cols = ['event', 'is_male', 'time', 'is_white', 'renal', 'cns', 'coagulation', 'cardiovascular']
        cols_standardize = list(set(df_full.columns.to_list()).symmetric_difference(skip_cols))
    elif data == 'hosp':
        df_full = pd.read_csv('data/MIMIC/MIMIC_IV_hosp_failure.csv')
        cols_standardize = []
    df = df_full.drop(df_full[df_full.event == 0].index)
    df.reset_index(drop=True, inplace=True)
    censor_time = make_synth_censor(censor_dist, df, df_full)
    censor_time = np.round(censor_time).astype(int)
    df = combine_data_with_censor(df, censor_time)
    df[cols_standardize] = df[cols_standardize].apply(lambda x: (x - x.mean()) / x.std())
    compare_km_curves(df_full, df, save_fig='Figs/MIMIC_IV_{}_{}.png'.format(data, censor_dist))
    return df


def make_semi_synth_metabric(
        censor_dist: str = 'Uniform'
) -> pd.DataFrame:
    df_full = metabric.read_df().rename(columns={"duration": "time"})
    cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
    df = df_full.drop(df_full[df_full.event == 0].index)
    df.reset_index(drop=True, inplace=True)
    df.time = df.time.round().astype(int)
    censor_time = make_synth_censor(censor_dist, df, df_full)
    censor_time = np.round(censor_time).astype(int)
    df = combine_data_with_censor(df, censor_time)
    df[cols_standardize] = df[cols_standardize].apply(lambda x: (x - x.mean()) / x.std())
    compare_km_curves(df_full, df, save_fig='Figs/Metabric_{}.png'.format(censor_dist))
    return df


def make_semi_synth_support(
        censor_dist: str = 'Uniform'
) -> pd.DataFrame:
    """Downloads and preprocesses the SUPPORT dataset from [1]_.

    The missing values are filled using either the recommended
    standard values, the mean (for continuous variables) or the mode
    (for categorical variables).
    Refer to the dataset description at
    https://biostat.app.vumc.org/wiki/Main/SupportDesc for more information.

    Returns
    -------
    pd.DataFrame
        DataFrame with processed covariates for one patient in each row.

    References
    ----------
    ..[1] W. A. Knaus et al., ‘The SUPPORT Prognostic Model: Objective Estimates of Survival
    for Seriously Ill Hospitalized Adults’, Ann Intern Med, vol. 122, no. 3, p. 191, Feb. 1995.
    """
    url = "https://biostat.app.vumc.org/wiki/pub/Main/DataSets/support2csv.zip"

    # Remove other target columns and other model predictions
    cols_to_drop = ["hospdead", "slos", "charges", "totcst", "totmcst", "avtisst", "sfdm2",
                    "adlp", "adls", "dzgroup",  # "adlp", "adls", and "dzgroup" were used in other preprocessing steps,
                    # see https://github.com/autonlab/auton-survival/blob/master/auton_survival/datasets.py
                    "sps", "aps", "surv2m", "surv6m", "prg2m", "prg6m", "dnr", "dnrday", "hday"]

    # `death` is the overall survival event indicator
    # `d.time` is the time to death from any cause or censoring
    df_full = (pd.read_csv(url).drop(cols_to_drop, axis=1).rename(columns={"d.time": "time", "death": "event"}))
    df_full["event"] = df_full["event"].astype(int)
    df_full["ca"] = (df_full["ca"] == "metastatic").astype(int)

    # use recommended default values from official dataset description ()
    # or mean (for continuous variables)/mode (for categorical variables) if not given
    fill_vals = {
        "alb": 3.5,
        "pafi": 333.3,
        "bili": 1.01,
        "crea": 1.01,
        "bun": 6.51,
        "wblc": 9,
        "urine": 2502,
        "edu": df_full["edu"].mean(),
        "ph": df_full["ph"].mean(),
        "glucose": df_full["glucose"].mean(),
        "scoma": df_full["scoma"].mean(),
        "meanbp": df_full["meanbp"].mean(),
        "hrt": df_full["hrt"].mean(),
        "resp": df_full["resp"].mean(),
        "temp": df_full["temp"].mean(),
        "sod": df_full["sod"].mean(),
        "income": df_full["income"].mode()[0],
        "race": df_full["race"].mode()[0],
    }
    df_full = df_full.fillna(fill_vals)

    df_full.sex.replace({'male': 1, 'female': 0}, inplace=True)
    df_full.income.replace({'under $11k': 0, '$11-$25k': 1, '$25-$50k': 2, '>$50k': 3}, inplace=True)
    skip_cols = ['event', 'sex', 'time', 'dzclass', 'race', 'diabetes', 'dementia', 'ca']
    cols_standardize = list(set(df_full.columns.to_list()).symmetric_difference(skip_cols))

    # one-hot encode categorical variables
    onehot_cols = ["dzclass", "race"]
    df_full = pd.get_dummies(df_full, columns=onehot_cols, drop_first=True)
    df_full = df_full.rename(columns={"dzclass_COPD/CHF/Cirrhosis": "dzclass_COPD"})

    # Drop all censored patients
    df = df_full.drop(df_full[df_full.event == 0].index)
    df.reset_index(drop=True, inplace=True)
    censor_time = make_synth_censor(censor_dist, df, df_full)
    censor_time = np.round(censor_time).astype(int)
    df = combine_data_with_censor(df, censor_time)
    df[cols_standardize] = df[cols_standardize].apply(lambda x: (x - x.mean()) / x.std())
    compare_km_curves(df_full, df, save_fig='Figs/SUPPORT_{}.png'.format(censor_dist))
    return df


def make_semi_synth_gbm(
        censor_dist: str = 'Uniform'
) -> pd.DataFrame:
    cols_standardize = ['years_to_birth', 'date_of_initial_pathologic_diagnosis', 'karnofsky_performance_score']
    df, df_full = get_uncensor_gbm()
    df.time = df.time.astype(int)
    censor_time = make_synth_censor(censor_dist, df, df_full)
    censor_time = np.round(censor_time).astype(int)
    df = combine_data_with_censor(df, censor_time)
    df[cols_standardize] = df[cols_standardize].apply(lambda x: (x - x.mean()) / x.std())
    compare_km_curves(df_full, df, save_fig='Figs/gbm_{}.png'.format(censor_dist))
    return df


def combine_data_with_censor(
        df: pd.DataFrame,
        censor_time: np.ndarray,
) -> pd.DataFrame:
    event_status = df.event.values
    true_times = df.time.values
    times = np.copy(true_times)
    event_status[censor_time < true_times] = 0
    times[event_status == 0] = censor_time[event_status == 0]
    # Combine the dataset
    df.drop(columns=["time", "event"], inplace=True)
    df["time"] = times
    df["event"] = event_status
    df["true_time"] = true_times
    df = df[df.time != 0]  # Drop all patients with censor time 0
    df.reset_index(drop=True, inplace=True)
    return df


def get_uncensor_gbm() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_full = pd.read_csv("data/GBM/GBM.clin.merged.picked.csv").rename(columns={"delta": "event"})
    df_full.drop(columns=["Composite Element REF", "tumor_tissue_site"], inplace=True)  # Columns with only one value
    df_full = df_full[df_full.time.notna()]  # Unknown censor/event time
    df_full.reset_index(drop=True, inplace=True)
    # Preprocess and fill missing values
    df_full.gender.replace({'male': 1, 'female': 0}, inplace=True)
    df_full.radiation_therapy.replace({'yes': 1, 'no': 0}, inplace=True)
    df_full.ethnicity.replace({'not hispanic or latino': 0, 'hispanic or latino': 1}, inplace=True)
    # one-hot encode categorical variables
    onehot_cols = ["histological_type", "race"]
    df_full = pd.get_dummies(df_full, columns=onehot_cols, drop_first=True)
    fill_vals = {
        "radiation_therapy": df_full["radiation_therapy"].median(),
        "karnofsky_performance_score": df_full["karnofsky_performance_score"].median(),
        "ethnicity": df_full["ethnicity"].median()
    }
    df_full = df_full.fillna(fill_vals)
    df_full.columns = df_full.columns.str.replace(" ", "_")
    # Drop all censored patients
    df = df_full.drop(df_full[df_full.event == 0].index)
    df.reset_index(drop=True, inplace=True)
    return df, df_full


def make_synth_censor(
        censor_dist: str,
        df_event: pd.DataFrame,
        df_all: pd.DataFrame
) -> np.ndarray:
    """
    Build synthetic censoring times
    :param censor_dist: type of censor distribution
    :param df_event: dataframe with all event patients
    :param df_all: dataframe with all patients
    :return: synthetic censoring times
    """
    if censor_dist == "uniform":
        censor_times = np.random.uniform(low=0, high=df_event.time.max(), size=df_event.shape[0])
    elif censor_dist == "uniform_truc":  # Uniform distribution with administrative censoring
        censor_times = np.random.uniform(low=0, high=df_event.time.max(), size=df_event.shape[0])
        censor_times[censor_times > df_event.time.median()] = df_event.time.median()
    elif censor_dist == "exponential":
        censor_times = np.random.exponential(scale=df_event.time.std(), size=df_event.shape[0])
    elif censor_dist in ["original_ind", "GBM"]:
        # Use original censor distribution or the external GBM censor distribution from the dataset.
        # This assumes the censor time C is independent of the features X
        if censor_dist == "GBM":
            # Use the GBM censor distribution to substitute the original censor distribution
            _, new_df_all = get_uncensor_gbm()
            new_df_all.time = new_df_all.time * df_all.time.max() / new_df_all.time.max()
            df_all = new_df_all
        inverse_prob_censor_est = KaplanMeier(df_all.time.values, 1 - df_all.event.values)
        uniq_times = inverse_prob_censor_est.survival_times
        censor_pdf = inverse_prob_censor_est.probability_dens / inverse_prob_censor_est.probability_dens.sum()
        censor_times = np.random.choice(uniq_times, size=df_event.shape[0], p=censor_pdf)
    elif censor_dist == "original_dep":
        # Use original censoring distribution from the dataset.
        # This assumes the censor time C is dependent of the features X. But C is independent of E given X.
        df_all_copy = df_all.copy()  # Make a copy to avoid changing the original dataset
        df_all_copy.event = 1 - df_all_copy.event
        cph = CoxPHFitter(penalizer=0.0001)
        cph.fit(df_all_copy, duration_col='time', event_col='event')
        censor_curves = cph.predict_survival_function(df_event)
        uniq_times = censor_curves.index.values
        censor_cdf = 1 - censor_curves.values.T
        censor_cdf_extra = np.ones((censor_cdf.shape[0], censor_cdf.shape[1] + 1))
        censor_cdf_extra[:, :-1] = censor_cdf
        censor_pdf = np.diff(censor_cdf_extra, axis=1)
        censor_pdf /= censor_pdf.sum(axis=1)[:, None]
        censor_times = np.empty(censor_pdf.shape[0])
        for i in range(censor_pdf.shape[0]):
            censor_times[i] = np.random.choice(uniq_times, p=censor_pdf[i, :])
    else:
        raise ValueError(f"Unknown censor distribution: {censor_dist}")
    return censor_times


def make_semi_synth_data(
        dataset: str,
        censor_dist: str = 'Uniform'
) -> pd.DataFrame:
    if dataset == "GBM":
        return make_semi_synth_gbm(censor_dist)
    elif dataset == "SUPPORT":
        return make_semi_synth_support(censor_dist)
    elif dataset == "Metabric":
        return make_semi_synth_metabric(censor_dist)
    elif dataset == "MIMIC-IV":
        return make_semi_synth_mimiciv(data='icu', censor_dist=censor_dist)
    elif dataset == "MIMIC-IV_hosp":
        return make_semi_synth_mimiciv(data='hosp', censor_dist=censor_dist)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


if __name__ == "__main__":
    data = make_semi_synth_data('GBM', censor_dist='exponential')
