from __future__ import annotations

import torch
import math
import numpy as np

from .regression import mean_squared_error_metric, spearman_metric

"""
===============================
Survival
===============================
"""


def survival_c_index_metric(target, pred, event_observed):
    from lifelines.utils import concordance_index

    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    event_observed = (
        torch.tensor(event_observed)
        if not torch.is_tensor(event_observed)
        else event_observed
    )
    try:
        if event_observed.float().mean() < 1.0:
            return torch.tensor(
                concordance_index(target, pred, event_observed=event_observed)
            )
        else:
            return torch.tensor(concordance_index(target, pred))
    except:
        print(
            f"Is nan -- target: {torch.isnan(target).any()}, pred: {torch.isnan(pred).any()}, event_observed: {torch.isnan(event_observed).any()}"
        )
        return 0.5


def survival_mse_uncensored(target, pred, event_observed):
    # MSE for all events that were observed
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred

    # 'censoring' variable is True if an event was observed, i.e. was not censored
    event_observed = (
        torch.tensor(event_observed)
        if not torch.is_tensor(event_observed)
        else event_observed
    )

    return mean_squared_error_metric(
        target[event_observed == True], pred[event_observed == True]
    )


def survival_spearman_uncensored(target, pred, event_observed):
    # MSE for all events that were observed
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred

    # 'censoring' variable is True if an event was observed, i.e. was not censored
    event_observed = (
        torch.tensor(event_observed)
        if not torch.is_tensor(event_observed)
        else event_observed
    )

    if len(pred[event_observed == True]) <= 1:
        return torch.tensor(0.0)
    elif pred[event_observed == True].std() == 0:
        # Spearman metric is undefined if all values are the same
        return torch.tensor(0.0)
    elif target[event_observed == True].std() == 0:
        # Spearman metric is undefined if all values are the same
        return torch.tensor(0.0)

    return spearman_metric(target[event_observed == True], pred[event_observed == True])


def survival_censored_accuracy(target, pred, event_observed):
    # How many censored patients were
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred

    # 'censoring' variable is True if an event was observed, i.e. was not censored
    event_observed = (
        torch.tensor(event_observed)
        if not torch.is_tensor(event_observed)
        else event_observed
    )

    if len(pred[event_observed == False]) == 0:
        return torch.tensor(0.0)

    return (
        (pred[event_observed == False] > target[event_observed == False]).float().mean()
    )


def get_risk_at_times(times_to_evaluate, estimate):
    from sksurv.functions import StepFunction

    if isinstance(estimate, np.ndarray) and isinstance(estimate[0], StepFunction):
        try:
            return np.row_stack([chf(times_to_evaluate) for chf in estimate])
        except ValueError as e:
            print(e)
            return np.zeros((len(estimate), len(times_to_evaluate)))
    elif isinstance(estimate, dict):
        # We have dict with "borders" and "buckets" (probabilities)
        # need to cumulate risk across buckets and project to the times_to_evaluate
        cum_sum = np.cumsum(estimate["buckets"], axis=1)
        estimate = -np.array(
            [
                np.interp(
                    np.array(times_to_evaluate),
                    estimate["borders"][1:].numpy(),
                    cum_sum[i],
                )
                for i in range(len(estimate["buckets"]))
            ]
        )
        return estimate
    else:
        return np.repeat(np.expand_dims(estimate, 1), len(times_to_evaluate), axis=1)


def get_times_to_evaluate(
    times_train, event_observed_train, times_test, event_observed_test
):
    times_train = (
        torch.tensor(times_train) if not torch.is_tensor(times_train) else times_train
    )
    times_test = (
        torch.tensor(times_test) if not torch.is_tensor(times_test) else times_test
    )

    return torch.unique(
        torch.quantile(
            times_test[event_observed_test == 1], torch.linspace(0.1, 0.9, 9)
        )
    ).tolist()


def survival_dynamic_cumulative_dynamic_auc(
    times_train,
    censoring_train,
    times_test,
    censoring_test,
    estimate,
    estimate_dynamic,
    use_dynamic_predictions=True,
):
    """
    Calculate the dynamic cumulative AUC for survival data.

    :param times_train: The times of the training data
    :param times_test: The times of the test data
    :param estimate: The estimate of the survival function
    :param times_to_evaluate: The times to evaluate the dynamic cumulative AUC at
    :return: The dynamic cumulative AUC
    """
    # TODO: Extend this to support multiple timepoints for the estimate

    from sksurv.metrics import cumulative_dynamic_auc

    times_to_evaluate = get_times_to_evaluate(
        times_train, censoring_train, times_test, censoring_test
    )

    y_train = np.array(
        list(zip(censoring_train, times_train)),
        dtype=[
            ("b", "bool"),
            ("a", "float"),
        ],
    )
    y_test = np.array(
        list(zip(censoring_test, times_test)),
        dtype=[
            ("b", "bool"),
            ("a", "float"),
        ],
    )
    y_all = np.concatenate([y_train, y_test])

    if use_dynamic_predictions:
        estimate = get_risk_at_times(times_to_evaluate, estimate_dynamic)

    try:
        return cumulative_dynamic_auc(y_all, y_test, -estimate, times_to_evaluate)[1]
    except ValueError as e:
        print(e)
        return 0.5


def survival_dynamic_cumulative_dynamic_auc_fixed_prediction(
    times_train, censoring_train, times_test, censoring_test, estimate, estimate_dynamic
):
    return survival_dynamic_cumulative_dynamic_auc(
        times_train,
        censoring_train,
        times_test,
        censoring_test,
        estimate,
        estimate_dynamic,
        use_dynamic_predictions=False,
    )


def survival_dynamic_integrated_brier_score(
    times_train, censoring_train, times_test, censoring_test, estimate, estimate_dynamic
):
    """
    Calculate the dynamic integrated Brier score for survival data.
    """

    from sksurv.metrics import integrated_brier_score

    times_to_evaluate = get_times_to_evaluate(
        times_train, censoring_train, times_test, censoring_test
    )

    y_train = np.array(
        list(zip(censoring_train, times_train)),
        dtype=[
            ("b", "bool"),
            ("a", "float"),
        ],
    )
    y_test = np.array(
        list(zip(censoring_test, times_test)),
        dtype=[
            ("b", "bool"),
            ("a", "float"),
        ],
    )
    y_all = np.concatenate([y_train, y_test])

    estimate = get_risk_at_times(times_to_evaluate, estimate_dynamic)

    try:
        return integrated_brier_score(y_all, y_test, -estimate, times_to_evaluate)
    except Exception as e:
        print(e)
        return np.inf


def survival_dynamic_top_k_odds_ratio(
    times_train,
    censoring_train,
    times_test,
    censoring_test,
    estimate,
    estimate_dynamic,
    use_dynamic_predictions=True,
    k=0.1,
):
    """
    Calculates the odds ratio risk metric for the top k% of patients compared to the remaining patients.

    :param target: The target tensor
    :param pred: The prediction tensor, higher values are assumed to be higher risk
    :param event_observed: The censoring tensor, True if an event was observed, False if the patient was censored
    :return:
    """
    times_to_evaluate = get_times_to_evaluate(
        times_train, censoring_train, times_test, censoring_test
    )

    # TODO: Odds ratio like other metrics doest make sense without a timepoint to evaluate at
    def top_k_odds_one_time_point(target, pred, event_observed, k, time):
        # How many censored patients were
        target = torch.tensor(target) if not torch.is_tensor(target) else target
        pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred

        target = target < time

        # 'censoring' variable is True if an event was observed, i.e. was not censored
        event_observed = (
            torch.tensor(event_observed)
            if not torch.is_tensor(event_observed)
            else event_observed
        )

        # Sort the prediction tensor and keep track of indices
        sorted_indices = torch.argsort(-pred, descending=True)
        sorted_target = target[sorted_indices].bool()
        sorted_event_observed = event_observed[sorted_indices].bool()
        sorted_censored = ~sorted_event_observed & sorted_target

        # Calculate k% threshold index
        k_percent_index = math.ceil(len(pred) * k)

        # Find event rates in the top k% and remaining
        top_k_event_rate = (
            sorted_target[:k_percent_index] & sorted_event_observed[:k_percent_index]
        ).sum().float() / (~sorted_censored[:k_percent_index]).sum().float()
        if torch.isnan(
            top_k_event_rate
        ):  # All samples were censored in top and there is no event
            return 0.0

        remaining_event_rate = (
            sorted_target[k_percent_index:] & sorted_event_observed[k_percent_index:]
        ).sum().float() / (~sorted_censored[k_percent_index:]).sum().float()

        # CAREFUL: We are comparing risks to the entire population, not just the remaining patients
        remaining_event_rate = (sorted_target & sorted_event_observed).sum().float() / (
            ~sorted_censored
        ).sum().float()

        if remaining_event_rate == 0:
            return 10000.0

        # Calculate odds ratio
        odds_ratio = top_k_event_rate / remaining_event_rate

        return odds_ratio.item()

    if use_dynamic_predictions:
        estimate = get_risk_at_times(times_to_evaluate, estimate_dynamic)
        odds_ratios = np.array(
            [
                top_k_odds_one_time_point(
                    times_test, estimate[:, i], censoring_test, k, time
                )
                for (i, time) in enumerate(times_to_evaluate)
            ]
        )
    else:
        odds_ratios = np.array(
            [
                top_k_odds_one_time_point(times_test, estimate, censoring_test, k, time)
                for time in times_to_evaluate
            ]
        )

    return odds_ratios.mean()


def survival_dynamic_top_10_odds_ratio(*args, **kwargs):
    return survival_dynamic_top_k_odds_ratio(*args, **kwargs, k=0.1)


def survival_dynamic_top_5_odds_ratio(*args, **kwargs):
    return survival_dynamic_top_k_odds_ratio(*args, **kwargs, k=0.05)


def survival_dynamic_top_1_odds_ratio(*args, **kwargs):
    return survival_dynamic_top_k_odds_ratio(*args, **kwargs, k=0.01)
