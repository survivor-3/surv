import math
import random

import numpy as np
import torch
from torch import nn

from tabpfn.utils import normalize_data
from tabpfn.datasets import SurvivalDataset
from tabpfn.priors.prior import QualityException


def set_censoring_time(
    returns, y, y_transformed, hyperparameters, device, single_eval_pos
):
    survival_censoring_competing = hyperparameters.get(
        "survival_censoring_competing", False
    )

    survival_censoring_global = hyperparameters.get("survival_censoring_global", "off")

    survival_censoring_independent = hyperparameters.get(
        "survival_censoring_independent", False
    )
    survival_censoring_global_fraction_all = hyperparameters.get(
        "survival_censoring_global_fraction_all", True
    )

    returns.settings[0]["survival_censoring_global"] = survival_censoring_global
    returns.settings[0][
        "survival_censoring_independent"
    ] = survival_censoring_independent
    returns.settings[0][
        "survival_censoring_global_fraction_all"
    ] = survival_censoring_global_fraction_all

    event_times_unobserved = y_transformed

    if survival_censoring_competing and returns.additional_x[0].shape[-1] > 0:
        idx = random.randint(0, returns.additional_x[0].shape[-1] - 1)
        additional_x = returns.additional_x[0][:, :, idx]

        returns.censoring_time = Survival().forward(
            additional_x.unsqueeze(-1).to(device), normalize_positions=single_eval_pos
        )

        if (returns.censoring_time < event_times_unobserved).all():
            returns.censoring_time = (
                (returns.censoring_time / returns.censoring_time.max())
                * event_times_unobserved.max()
                * (1 + random.random())
            )

        return

    num_samples, batch_size, N_out = y.shape[0], y.shape[1], y.shape[2]
    # TODO: No censoring outside of event interval
    # TODO: Survival distribution should be a mix of dists?

    returns.censoring_time = torch.ones(size=(num_samples, batch_size, N_out)) * 1000000

    if survival_censoring_global in ["off", "add"]:
        if (
            survival_censoring_independent
            or len(returns.additional_x) == 0
            or returns.additional_x[0].shape[-1] == 0
        ):
            time_min, time_max, time_mean = (
                event_times_unobserved.min(dim=0)[0],
                event_times_unobserved.max(dim=0)[0],
                event_times_unobserved.mean(dim=0),
            )

            # time is censoring time
            returns.censoring_time = (
                time_min
                + torch.abs(torch.randn(size=(num_samples, batch_size, N_out)))
                * (time_mean - time_min)
                * 5
                * random.random()
            )
        else:
            time_min, time_max, time_mean = (
                event_times_unobserved.min(dim=0)[0],
                event_times_unobserved.max(dim=0)[0],
                event_times_unobserved.mean(dim=0),
            )

            idx = random.randint(0, returns.additional_x[0].shape[-1] - 1)
            additional_x = returns.additional_x[0][:, :, idx]
            additional_x_min, additional_x_max = (
                additional_x.min(dim=0)[0],
                additional_x.max(dim=0)[0],
            )
            additional_x = additional_x - additional_x_min

            additional_x_mean = additional_x.mean(dim=0)
            additional_x = additional_x / additional_x_mean

            # time is censoring time
            returns.censoring_time = (
                time_min
                + torch.abs(torch.randn(size=(num_samples, batch_size, N_out)))
                * (time_mean - time_min)
                * additional_x.unsqueeze(1)
                * 5
                * random.random()
            )

    if survival_censoring_global in ["add", "only", "continue"]:
        time_min, time_max, time_mean = (
            event_times_unobserved.min(dim=0)[0],
            event_times_unobserved.max(dim=0)[0],
            event_times_unobserved.mean(dim=0),
        )

        global_censoring_time = (
            torch.abs(torch.randn(size=(1, batch_size, N_out))) * time_mean * 5
        )

        if survival_censoring_global_fraction_all:
            global_censored_idx = torch.ones(size=(num_samples, batch_size, N_out)) == 1
        else:
            global_censored_idx = torch.rand(
                size=(num_samples, batch_size, N_out)
            ) < hyperparameters.get("survival_censoring_global_fraction", 0.5)

        if survival_censoring_global == "continue":
            # from threshold all samples are censored, but the censored time is different for each sample
            returns.censoring_time[global_censored_idx] = torch.max(
                global_censoring_time, event_times_unobserved[global_censored_idx]
            )
        else:
            # one global censoring time for all samples
            returns.censoring_time[global_censored_idx] = torch.min(
                returns.censoring_time[global_censored_idx],
                global_censoring_time,
            )


def set_survival_targets(
    returns, y, y_transformed, hyperparameters, device, single_eval_pos
):
    set_censoring_time(
        returns, y, y_transformed, hyperparameters, device, single_eval_pos
    )

    returns.risks = y.clone()
    event_times_unobserved = y_transformed

    returns.event_observed = event_times_unobserved < returns.censoring_time
    returns.event_times_unobserved = event_times_unobserved.clone()
    returns.event_times = torch.where(
        returns.event_observed,
        event_times_unobserved,  # True, If an event occured
        returns.censoring_time,  # False, If no event occured, censoring time
    )

    if hyperparameters.get("use_censoring_loss", False):
        # Predicting the censored times as y labels
        returns.y = returns.event_times.clone()
    else:
        # Predicting true event times as y labels and feeding the censored times as train labels
        returns.y = returns.event_times_unobserved.clone()
        returns.y[:single_eval_pos] = returns.event_times[:single_eval_pos].clone()

    mean = (
        returns.y[:single_eval_pos].min()
        if hyperparameters.get("survival_normalize_with_min", True)
        else None
    )
    clip = 10000
    # Normalizing on the train y values
    returns.y, (data_mean, data_std) = normalize_data(
        returns.y,
        normalize_positions=single_eval_pos,
        mean=mean,
        return_scaling=True,
        clip=clip,
    )
    returns.event_times_unobserved = normalize_data(
        returns.event_times_unobserved,
        normalize_positions=-1,
        mean=data_mean,
        std=data_std,
        clip=clip,
    )
    returns.event_times = normalize_data(
        returns.event_times,
        normalize_positions=-1,
        mean=data_mean,
        std=data_std,
        clip=clip,
    )
    returns.censoring_time = normalize_data(
        returns.censoring_time,
        normalize_positions=-1,
        mean=data_mean,
        std=data_std,
        clip=clip,
    )

    # Mask out censoring indicator for the evaluation points
    # We pass the censoring indicator via x and not y, because appending to y would mean we are trying to make
    #   predictions of the censoring indicator.
    masked_events = returns.event_observed.clone().float()
    masked_events[single_eval_pos:] = SurvivalDataset.get_missing_event_indicator()
    returns.event_observed_masked_for_train = masked_events.squeeze(-1)

    if (
        returns.event_observed[:single_eval_pos].sum() <= 6
    ):  # Not enough uncensored samples in the training set
        raise QualityException(
            "Too few uncensored samples in train, skipping"
        )  # raising leads to skipping the batch
        returns.y[:] = np.nan  # Makes the batch invalid
    elif (
        returns.event_observed[single_eval_pos:].sum() <= 6
    ):  # Not enough uncensored samples in the test set
        raise QualityException("Too few uncensored samples in test, skipping")
        returns.y[:] = np.nan  # Makes the batch invalid
    elif (returns.y >= 1000).any():
        raise QualityException("Some ys too large, skipping")
        returns.y[:] = np.nan
    elif returns.y[single_eval_pos:].max() > returns.y[:single_eval_pos].max() * 10:
        raise QualityException(
            "Test ys too large compared to train, skipping (i.e. censored samples true event times much larger than uncensored)"
        )
        returns.y[:] = np.nan

    return returns


class Survival(nn.Module):
    """
    Based on: https://github.com/dsciencelabs/survivalmodel_py/blob/0695c3bb063092831d431924af76618195363cdd/pysurvival/models/simulations.py

    """

    def __init__(
        self, survival_distribution=None, survival_distribution_mixed_sampler=None
    ):
        if survival_distribution is None:
            survival_distribution = {
                "name": "exp",
                "exp_alpha": lambda: 0.1,
            }
        self.survival_distribution = survival_distribution
        self.survival_distribution_mixed_sampler = survival_distribution_mixed_sampler
        super().__init__()

    @staticmethod
    def gompertz_icdf(u, eta, b):
        """
        Inverse CDF of the Gompertz distribution.

        Args:
        u (torch.Tensor): random variables in the range (0, 1).
        eta (float): Shape parameter of the Gompertz distribution.
        b (float): Scale parameter of the Gompertz distribution.

        Returns:
        torch.Tensor: Gompertz distributed random variables.
        """
        return (1 / eta) * torch.log(1 - torch.log(u) / b)

    @staticmethod
    def log_normal_icdf(u, mu, sigma):
        """
        Inverse CDF of the log-normal distribution.

        Args:
        u (torch.Tensor): random variables in the range (0, 1).
        mu (float): Mean of the log-normal distribution.
        sigma (float): Standard deviation of the log-normal distribution.

        Returns:
        torch.Tensor: Log-normal distributed random variables.
        """
        return torch.exp(mu + sigma * math.sqrt(2) * torch.erfinv(2 * u - 1))

    @staticmethod
    def exponential_icdf(u, alpha):
        """
        Inverse CDF of the exponential distribution.

        Args:
        u (torch.Tensor): random variables in the range (0, 1).
        alpha (float): Shape parameter of the exponential distribution.

        Returns:
        torch.Tensor: Exponential distributed random variables.
        """
        return -torch.log(1 - u) / alpha

    @staticmethod
    def normal_to_uniform(y):
        # Calculate CDF of standard normal distribution
        phi = lambda x: 0.5 * (1 + torch.erf(x / np.sqrt(2)))
        standard_normal_cdf = phi(y)

        # Scale and shift to [0, 1]
        risks = (standard_normal_cdf - torch.min(standard_normal_cdf)) / (
            torch.max(standard_normal_cdf) - torch.min(standard_normal_cdf)
        )
        risks = torch.clamp(
            risks, 0.00001 + 0.0001 * random.random(), 0.999 + 0.0009 * random.random()
        )

        return risks

    def forward(self, y, normalize_positions=-1):
        # risks is S, B, N_Out
        if self.survival_distribution["name"].lower() == "mixed":
            samples_list = []
            for i in range(10):
                dist_name = self.survival_distribution_mixed_sampler()
                dist = Survival(self.survival_distribution)
                dist.survival_distribution["name"] = dist_name
                risks = dist(y, normalize_positions=normalize_positions)
                samples_list += [risks]
            samples = torch.stack(samples_list, dim=-1).mean(dim=-1)
        elif self.survival_distribution["name"].lower() == "gompertz":
            risks = Survival.normal_to_uniform(y)

            ## Transform the standard normal x to a uniform distribution u in the range (0, 1)
            # risks = torch.clamp(torch.sigmoid(y), min=0.001, max=0.999)

            # Gompertz distribution parameters
            eta = self.survival_distribution["gompertz_eta"]()
            b = self.survival_distribution["gompertz_b"]()

            # Transform u to Gompertz distributed random variables
            samples = Survival.gompertz_icdf(risks, eta, b)
        elif self.survival_distribution["name"].lower() == "log_normal":
            risks = Survival.normal_to_uniform(y)

            # Log-normal distribution parameters
            mu = self.survival_distribution["lognormal_mu"]()
            sigma = self.survival_distribution["lognormal_sigma"]()

            # Transform u to log-normal distributed random variables
            samples = Survival.log_normal_icdf(risks, mu, sigma)
        elif self.survival_distribution["name"].lower() == "exp":
            risks = Survival.normal_to_uniform(y)

            # Exponential distribution parameters
            alpha = self.survival_distribution["exp_alpha"]()

            # Transform u to exponential distributed random variables
            samples = Survival.exponential_icdf(risks, alpha)
        else:
            raise NotImplementedError(
                f"Unknowm survival distribution: {self.survival_distribution['name']}"
            )

        # Normalize values to start at 0 and have a std of 1
        out = normalize_data(
            samples, normalize_positions=normalize_positions, std_only=True, clip=-1
        )

        return out


class SurvivalOld(nn.Module):
    """
    Based on: https://github.com/dsciencelabs/survivalmodel_py/blob/0695c3bb063092831d431924af76618195363cdd/pysurvival/models/simulations.py

    """

    def __init__(self, survival_distribution=None):
        if survival_distribution is None:
            survival_distribution = {"name": "exp", "alpha": 1}
        self.survival_distribution = survival_distribution
        super().__init__()

    def time_function(self, BX):
        """
        Calculating the survival times based on the given distribution
        T = H^(-1)( -log(U)/risk_score ), where:
            * H is the cumulative baseline hazard function
                (H^(-1) is the inverse function)
            * U is a random variable uniform - Uni[0,1].
        The method is inspired by https://gist.github.com/jcrudy/10481743
        """

        # Calculating scale coefficient using the features
        num_samples, batch_size, N_out = BX.shape[0], BX.shape[1], BX.shape[2]

        # Generating random uniform variables
        U = torch.rand(size=(num_samples, batch_size, N_out), device=BX.device)
        lambda_exp_BX = torch.exp(BX) * self.survival_distribution["alpha"]

        # Exponential
        if self.survival_distribution["name"].lower().startswith("exp"):
            return -torch.log(U) / (lambda_exp_BX)
        # Weibull
        elif self.survival_distribution.lower().startswith("wei"):
            self.survival_distribution = "Weibull"
            return np.power(-np.log(U) / (lambda_exp_BX), 1.0 / self.beta)

        # Gompertz
        elif self.survival_distribution.lower().startswith("gom"):
            self.survival_distribution = "Gompertz"
            return (1.0 / self.beta) * np.log(
                1 - self.beta * np.log(U) / (lambda_exp_BX)
            )

        # Log-Logistic
        elif "logistic" in self.survival_distribution.lower():
            self.survival_distribution = "Log-Logistic"
            return np.power(U / (1.0 - U), 1.0 / self.beta) / (lambda_exp_BX)

        # Log-Normal
        elif "normal" in self.survival_distribution.lower():
            self.survival_distribution = "Log-Normal"
            W = np.random.normal(0, 1, num_samples)
            return lambda_exp_BX * np.exp(self.beta * W)

    def forward(self, y, normalize_positions=-1):
        # risks is S, B, N_Out
        # TODO: Softplus change?
        risks = torch.nn.functional.softplus(y, beta=0.5)

        event_times = self.time_function(risks)

        return event_times
