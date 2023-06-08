import torch
import math
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from typing import Optional, Union, List, Tuple
import matplotlib.pyplot as plt
from lifelines.statistics import logrank_test

from skmultilearn.model_selection import iterative_train_test_split

Numeric = Union[float, int, bool]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]


class KaplanMeier:
    """
    This class is borrowed from survival_evaluation package.
    """
    def __init__(self, event_times, event_indicators):
        self.event_times = event_times
        self.event_indicators = event_indicators

        index = np.lexsort((event_indicators, event_times))
        unique_times = np.unique(event_times[index], return_counts=True)
        self.survival_times = unique_times[0]
        population_count = np.flip(np.flip(unique_times[1]).cumsum())

        event_counter = np.append(0, unique_times[1].cumsum()[:-1])
        event_ind = list()
        for i in range(np.size(event_counter[:-1])):
            event_ind.append(event_counter[i])
            event_ind.append(event_counter[i + 1])
        event_ind.append(event_counter[-1])
        event_ind.append(len(event_indicators))
        events = np.add.reduceat(np.append(event_indicators[index], 0), event_ind)[::2]

        self.survival_probabilities = np.empty(population_count.size)
        survival_probability = 1
        counter = 0
        for population, event_num in zip(population_count, events):
            survival_probability *= 1 - event_num / population
            self.survival_probabilities[counter] = survival_probability
            counter += 1
        self.cumulative_dens = 1 - self.survival_probabilities
        self.probability_dens = np.diff(np.append(self.cumulative_dens, 1))

    def predict(self, prediction_times: np.array):
        probability_index = np.digitize(prediction_times, self.survival_times)
        probability_index = np.where(
            probability_index == self.survival_times.size + 1,
            probability_index - 1,
            probability_index,
        )
        probabilities = np.append(1, self.survival_probabilities)[probability_index]

        return probabilities


def compare_km_curves(df1, df2, intervals=None, save_fig=None):
    results = logrank_test(df1.time.values, df2.time.values, df1.event.values, df2.event.values)

    event_times_1 = df1.time.values[df1.event.values == 1]
    censor_times_1 = df1.time.values[df1.event.values == 0]
    event_times_2 = df2.time.values[df2.event.values == 1]
    censor_times_2 = df2.time.values[df2.event.values == 0]
    if intervals is None:
        intervals = 21  # 20 bins
    bins = np.linspace(0, round(df1.time.max()), intervals)

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    ax0.hist([event_times_1, censor_times_1], bins=bins, histtype='bar', stacked=True)
    ax0.legend(['Event times', 'Censor Times'])
    ax0.set_title("Event/Censor Time Histogram")

    km_estimator = KaplanMeier(df1.time.values, df1.event.values)
    ax1.plot(km_estimator.survival_times, km_estimator.survival_probabilities, linewidth=3)
    ax1.set_title("Kaplan-Meier Curve")
    ax1.set_ylim([0, 1])
    xmin, xmax = ax1.get_xlim()

    ax2.hist([event_times_2, censor_times_2], bins=bins, histtype='bar', stacked=True)
    ax2.legend(['Event times', 'Censor Times'])
    # ax2.set_title("Event/Censor Times Histogram")

    km_estimator = KaplanMeier(df2.time.values, df2.event.values)
    ax3.plot(km_estimator.survival_times, km_estimator.survival_probabilities, linewidth=3)
    # ax3.set_title("Kaplan-Meier Curve")
    ax3.set_ylim([0, 1])
    ax3.set_xlim([xmin, xmax])

    # fig.set_size_inches(12, 12)
    plt.suptitle('Logrank Test: p-value = {:.5f}'.format(results.p_value))
    plt.setp(ax0, xlabel='Time', ylabel='Counts')
    plt.setp(ax1, xlabel='Time', ylabel='Probabilities')
    plt.setp(ax2, xlabel='Time', ylabel='Counts')
    plt.setp(ax3, xlabel='Time', ylabel='Probabilities')
    # plt.show()
    if save_fig is not None:
        fig.savefig(save_fig, dpi=300)


def plot_time_hist(event_censor_times, event_indicators, intervals=None, save_fig=None):
    # Plot the event/censor times histogram and the kaplan meier curve for a given dataset
    event_times = event_censor_times[event_indicators == 1]
    censor_times = event_censor_times[event_indicators == 0]
    if intervals is None:
        intervals = 21  # 20 bins
    bins = np.linspace(0, round(event_times.max()), intervals)

    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)

    ax0.hist([event_times, censor_times], bins=bins, histtype='bar', stacked=True)
    ax0.legend(['Event times', 'Censor Times'])
    ax0.set_title("Event/Censor Times Histogram")

    km_estimator = KaplanMeier(event_censor_times, event_indicators)
    ax1.plot(km_estimator.survival_times, km_estimator.survival_probabilities, linewidth=3)
    ax1.set_title("Kaplan-Meier Curve")
    ax1.set_ylim([0, 1])
    fig.set_size_inches(14, 7)
    plt.setp(ax0, xlabel='Time', ylabel='Counts')
    plt.setp(ax1, xlabel='Time', ylabel='Probabilities')
    plt.show()

    if save_fig is not None:
        fig.savefig(save_fig, dpi=300)


def extract_survival(dataset):
    """Extracts the feature, survival time, and event/censor bits of the patients in the dataset"""
    return (torch.tensor(dataset.drop(["time", "event"], axis=1).values, dtype=torch.float),
            torch.tensor(dataset["time"].values, dtype=torch.float),
            torch.tensor(dataset["event"].values, dtype=torch.float))


def is_monotonic(
        array: Union[torch.Tensor, np.ndarray, list]
):
    return (all(array[i] <= array[i + 1] for i in range(len(array) - 1)) or
            all(array[i] >= array[i + 1] for i in range(len(array) - 1)))


def compute_unique_counts(
        event: torch.Tensor,
        time: torch.Tensor,
        order: Optional[torch.Tensor] = None):
    """Count right censored and uncensored samples at each unique time point.

    Parameters
    ----------
    event : array
        Boolean event indicator.

    time : array
        Survival time or time of censoring.

    order : array or None
        Indices to order time in ascending order.
        If None, order will be computed.

    Returns
    -------
    times : array
        Unique time points.

    n_events : array
        Number of events at each time point.

    n_at_risk : array
        Number of samples that have not been censored or have not had an event at each time point.

    n_censored : array
        Number of censored samples at each time point.
    """
    n_samples = event.shape[0]

    if order is None:
        order = torch.argsort(time)

    uniq_times = torch.empty(n_samples, dtype=time.dtype, device=time.device)
    uniq_events = torch.empty(n_samples, dtype=torch.int, device=time.device)
    uniq_counts = torch.empty(n_samples, dtype=torch.int, device=time.device)

    i = 0
    prev_val = time[order[0]]
    j = 0
    while True:
        count_event = 0
        count = 0
        while i < n_samples and prev_val == time[order[i]]:
            if event[order[i]]:
                count_event += 1

            count += 1
            i += 1

        uniq_times[j] = prev_val
        uniq_events[j] = count_event
        uniq_counts[j] = count
        j += 1

        if i == n_samples:
            break

        prev_val = time[order[i]]

    uniq_times = uniq_times[:j]
    uniq_events = uniq_events[:j]
    uniq_counts = uniq_counts[:j]
    n_censored = uniq_counts - uniq_events

    # offset cumulative sum by one
    total_count = torch.cat([torch.tensor([0], device=uniq_counts.device), uniq_counts], dim=0)
    n_at_risk = n_samples - torch.cumsum(total_count, dim=0)

    return uniq_times, uniq_events, n_at_risk[:-1], n_censored


def baseline_hazard(
        logits: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Calculate the baseline cumulative hazard function and baseline survival function using Breslow estimator
    :param logits: logit outputs calculated from the Cox-based network using training data.
    :param time: Survival time of training data.
    :param event: Survival indicator of training data.
    :return:
    uniq_times: time bins correspond of the baseline hazard/survival.
    cum_baseline_hazard: cumulative baseline hazard
    baseline_survival: baseline survival curve.
    """
    risk_score = torch.exp(logits)
    order = torch.argsort(time)
    risk_score = risk_score[order]
    uniq_times, n_events, n_at_risk, _ = compute_unique_counts(event, time, order)

    divisor = torch.empty(n_at_risk.shape, dtype=torch.float, device=n_at_risk.device)
    value = torch.sum(risk_score)
    divisor[0] = value
    k = 0
    for i in range(1, len(n_at_risk)):
        d = n_at_risk[i - 1] - n_at_risk[i]
        value -= risk_score[k:(k + d)].sum()
        k += d
        divisor[i] = value

    assert k == n_at_risk[0] - n_at_risk[-1]

    hazard = n_events / divisor
    # Make sure the survival curve always starts at 1
    if 0 not in uniq_times:
        uniq_times = torch.cat([torch.tensor([0]).to(uniq_times.device), uniq_times], 0)
        hazard = torch.cat([torch.tensor([0]).to(hazard.device), hazard], 0)
    # TODO: torch.cumsum with cuda array will generate a non-monotonic array. Need to update when torch fix this bug
    # See issue: https://github.com/pytorch/pytorch/issues/21780
    cum_baseline_hazard = torch.cumsum(hazard.cpu(), dim=0).to(hazard.device)
    baseline_survival = torch.exp(- cum_baseline_hazard)
    if baseline_survival.isinf().any() or (not is_monotonic(baseline_survival)):
        print(f"Baseline survival contains \'inf\', need attention. \n"
              f"Baseline survival distribution: {baseline_survival}")
        last_zero = torch.where(baseline_survival == 0)[0][-1].item()
        baseline_survival[last_zero + 1:] = 0
    return uniq_times, hazard, cum_baseline_hazard, baseline_survival


def reformat_survival(
        dataset: pd.DataFrame,
        time_bins: NumericArrayLike
) -> (torch.Tensor, torch.Tensor):
    x = torch.tensor(dataset.drop(["time", "event"], axis=1).values, dtype=torch.float)
    y = encode_survival(dataset["time"].values, dataset["event"].values, time_bins)
    return x, y


def encode_survival(
        time: Union[float, int, NumericArrayLike],
        event: Union[int, bool, NumericArrayLike],
        bins: NumericArrayLike
) -> torch.Tensor:
    """Encodes survival time and event indicator in the format
    required for MTLR training.

    For uncensored instances, one-hot encoding of binned survival time
    is generated. Censoring is handled differently, with all possible
    values for event time encoded as 1s. For example, if 5 time bins are used,
    an instance experiencing event in bin 3 is encoded as [0, 0, 0, 1, 0], and
    instance censored in bin 2 as [0, 0, 1, 1, 1]. Note that an additional
    'catch-all' bin is added, spanning the range `(bins.max(), inf)`.

    Parameters
    ----------
    time
        Time of event or censoring.
    event
        Event indicator (0 = censored).
    bins
        Bins used for time axis discretisation.

    Returns
    -------
    torch.Tensor
        Encoded survival times.
    """
    # TODO this should handle arrays and (CUDA) tensors
    if isinstance(time, (float, int, np.ndarray)):
        time = np.atleast_1d(time)
        time = torch.tensor(time)
    if isinstance(event, (int, bool, np.ndarray)):
        event = np.atleast_1d(event)
        event = torch.tensor(event)

    if isinstance(bins, np.ndarray):
        bins = torch.tensor(bins)

    try:
        device = bins.device
    except AttributeError:
        device = "cpu"

    time = np.clip(time, 0, bins.max())
    # add extra bin [max_time, inf) at the end
    y = torch.zeros((time.shape[0], bins.shape[0] + 1),
                    dtype=torch.float,
                    device=device)
    # For some reason, the `right` arg in torch.bucketize
    # works in the _opposite_ way as it does in numpy,
    # so we need to set it to True
    bin_idxs = torch.bucketize(time, bins, right=True)
    for i, (bin_idx, e) in enumerate(zip(bin_idxs, event)):
        if e == 1:
            y[i, bin_idx] = 1
        else:
            y[i, bin_idx:] = 1
    return y.squeeze()


def make_time_bins(
        times: NumericArrayLike,
        num_bins: Optional[int] = None,
        use_quantiles: bool = True,
        event: Optional[NumericArrayLike] = None,
        add_last_time: Optional[bool] = False
) -> torch.Tensor:
    """Creates the bins for survival time discretisation.

    By default, sqrt(num_observation) bins corresponding to the quantiles of
    the survival time distribution are used, as in https://github.com/haiderstats/MTLR.

    Parameters
    ----------
    times
        Array or tensor of survival times.
    num_bins
        The number of bins to use. If None (default), sqrt(num_observations)
        bins will be used.
    use_quantiles
        If True, the bin edges will correspond to quantiles of `times`
        (default). Otherwise, generates equally-spaced bins.
    event
        Array or tensor of event indicators. If specified, only samples where
        event == 1 will be used to determine the time bins.
    add_last_time
        If True, the last time bin will be added to the end of the time bins.
    Returns
    -------
    torch.Tensor
        Tensor of bin edges.
    """
    # TODO this should handle arrays and (CUDA) tensors
    if event is not None:
        times = times[event == 1]
    if num_bins is None:
        num_bins = math.ceil(math.sqrt(len(times)))
    if use_quantiles:
        # NOTE we should switch to using torch.quantile once it becomes
        # available in the next version
        bins = np.unique(np.quantile(times, np.linspace(0, 1, num_bins)))
    else:
        bins = np.linspace(times.min(), times.max(), num_bins)
    bins = torch.tensor(bins, dtype=torch.float)
    if add_last_time:
        bins = torch.cat([bins, torch.tensor([times.max()])])
    return bins


def survival_stratified_cv(
        dataset: pd.DataFrame,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
        number_folds: int = 5
) -> list:
    event_times, event_indicators = event_times.tolist(), event_indicators.tolist()
    assert len(event_indicators) == len(event_times)

    indicators_and_times = list(zip(event_indicators, event_times))
    sorted_idx = [i[0] for i in sorted(enumerate(indicators_and_times), key=lambda v: (v[1][0], v[1][1]))]

    folds = [[sorted_idx[0]], [sorted_idx[1]], [sorted_idx[2]], [sorted_idx[3]], [sorted_idx[4]]]
    for i in range(5, len(sorted_idx)):
        fold_number = i % number_folds
        folds[fold_number].append(sorted_idx[i])

    training_sets = [dataset.drop(folds[i], axis='index').reset_index(drop=True) for i in range(number_folds)]
    testing_sets = [dataset.iloc[folds[i], :].reset_index(drop=True) for i in range(number_folds)]

    cross_validation_set = list(zip(training_sets, testing_sets))
    return cross_validation_set


def multilabel_train_test_split(x, y, test_size, random_state=None):
    """Iteratively stratified train/test split
    (Add random_state to scikit-multilearn iterative_train_test_split function)
    See this paper for details: https://link.springer.com/chapter/10.1007/978-3-642-23808-6_10
    """
    x, y = shuffle(x, y, random_state=random_state)
    x_train, y_train, x_test, y_test = iterative_train_test_split(x, y, test_size=test_size)
    return x_train, y_train, x_test, y_test


def survival_data_split(
        df: pd.DataFrame,
        stratify_colname: str = 'event',
        frac_train: float = 0.5,
        frac_val: float = 0.0,
        frac_test: float = 0.5,
        random_state: int = None
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    assert frac_train >= 0 and frac_val >= 0 and frac_test >= 0, "Check train validation test fraction."
    frac_sum = frac_train + frac_val + frac_test
    frac_train = frac_train / frac_sum
    frac_val = frac_val / frac_sum
    frac_test = frac_test / frac_sum

    x = df.values  # Contains all columns.
    columns = df.columns
    if stratify_colname == 'event':
        stra_lab = df[stratify_colname]
    elif stratify_colname == 'time':
        stra_lab = df[stratify_colname]
        bins = np.linspace(start=stra_lab.min(), stop=stra_lab.max(), num=20)
        stra_lab = np.digitize(stra_lab, bins, right=True)
    elif stratify_colname == "both":
        t = df["time"]
        bins = np.linspace(start=t.min(), stop=t.max(), num=20)
        t = np.digitize(t, bins, right=True)
        e = df["event"]
        stra_lab = np.stack([t, e], axis=1)
    else:
        raise ValueError("unrecognized stratify policy")

    x_train, _, x_temp, y_temp = multilabel_train_test_split(x, y=stra_lab, test_size=(1.0 - frac_train),
                                                             random_state=random_state)
    if frac_val == 0:
        x_val, x_test = [], x_temp
    else:
        x_val, _, x_test, _ = multilabel_train_test_split(x_temp, y=y_temp,
                                                          test_size=frac_test / (frac_val + frac_test),
                                                          random_state=random_state)
    df_train = pd.DataFrame(data=x_train, columns=columns)
    df_val = pd.DataFrame(data=x_val, columns=columns)
    df_test = pd.DataFrame(data=x_test, columns=columns)
    assert len(df) == len(df_train) + len(df_val) + len(df_test)
    return df_train, df_val, df_test
