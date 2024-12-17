import math

import random
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm
import copy

from . import (
    SyntheticDataBenchmark,
    FormulaBenchmark,
)

standard_N = 400
more_train_idx_func = lambda i, N: i % (N // 200) == 0 or i == 0 or i == N - 1
more_train_idx_func_with_gap = lambda i, N: (i % (N // 200) == 0) and not (
    N * 0.4 < i < N * 0.6
)
less_train_idx_func = lambda i, N: i % (N // 10) == 0 or i == 0 or i == N - 1
standard_train_idx_func = lambda i, N: i % (N // 40) == 0 or i == 0 or i == N - 1
generalization_idx_func = lambda i, N: (
    i < N * 0.9 and i > N * 0.1
)  # or i < 2 or i > N - 3
one_side_generalization_idx_func = lambda i, N: (i < N * 0.75)  # or i < 2 or i > N - 3

random_train_idx_func = lambda i, N: random.random() > 0.5

single_ordered_feature_sampler = lambda N: np.linspace(-1 / 2, 1 / 2, N)
random_feature_sampler = lambda N: np.random.rand(N) - 0.5
# single_ordered_feature_sampler = lambda N: np.sort(np.random.randn(N)) / 3


def get_multimodal_X_function(**kwargs):
    """
    This function is used to create regression task that is best predicted by outputting a multimodal distribution.
    The function looks akin to an X.

    :return:
    """

    def multimodal(a):
        return a if random.random() > 0.5 else -a

    benchmark = FormulaBenchmark(
        multimodal,
        {"a": single_ordered_feature_sampler},
        N=standard_N,
        train_idx_func=more_train_idx_func,
        **kwargs,
    )

    return benchmark


def get_modulo_function(**kwargs):
    def f(a):
        return a % 0.2

    benchmark = FormulaBenchmark(
        f,
        {"a": single_ordered_feature_sampler},
        N=standard_N,
        train_idx_func=standard_train_idx_func,
        **kwargs,
    )

    return benchmark


def get_step_function(**kwargs):
    def step_func(x):
        if x < -0.25:
            y = -0.4
        elif x < -0.0:
            y = -0.2
        elif x < 0.25:
            y = 0.2
        elif x <= 0.5:
            y = 0.4
        return y

    benchmark = FormulaBenchmark(
        lambda a: step_func(a),
        {"a": single_ordered_feature_sampler},
        N=standard_N,
        train_idx_func=standard_train_idx_func,
        **kwargs,
    )

    return benchmark


def get_power_function(p=2, **kwargs):
    offset = 0.5 if p == 2 else 0
    benchmark = FormulaBenchmark(
        lambda a: math.pow(a, p) * math.pow(2, p) - offset,
        {"a": single_ordered_feature_sampler},
        N=standard_N,
        train_idx_func=less_train_idx_func,
        **kwargs,
    )

    return benchmark


def get_abs_function(**kwargs):
    benchmark = FormulaBenchmark(
        lambda a: abs(a) * 2,
        {"a": single_ordered_feature_sampler},
        N=standard_N,
        train_idx_func=less_train_idx_func,
        **kwargs,
    )

    return benchmark


def get_sin_function(**kwargs):
    benchmark = FormulaBenchmark(
        lambda a: math.sin(a * 10) / 2,
        {"a": single_ordered_feature_sampler},
        N=standard_N,
        train_idx_func=standard_train_idx_func,
        **kwargs,
    )
    return benchmark


def get_sin_times_x_function(**kwargs):
    benchmark = FormulaBenchmark(
        lambda a: math.sin((a + 0.5) * (a + 0.5) * 20) / 2,
        {"a": single_ordered_feature_sampler},
        N=standard_N,
        train_idx_func=standard_train_idx_func,
        **kwargs,
    )
    return benchmark


def get_high_freq_function(f=100, n_samples=standard_N, **kwargs):
    benchmark = FormulaBenchmark(
        lambda a: math.sin((a + 0.5) * f) / 2,
        {"a": random_feature_sampler},
        N=n_samples,
        train_idx_func=random_train_idx_func,
        task_type="multiclass",
        **kwargs,
    )
    return benchmark


def get_high_freq_function(f=100, n_samples=standard_N, **kwargs):
    benchmark = FormulaBenchmark(
        lambda a: math.sin((a + 0.5) * f) / 2,
        {"a": random_feature_sampler},
        N=n_samples,
        train_idx_func=random_train_idx_func,
        task_type="multiclass",
        **kwargs,
    )
    return benchmark


def get_high_freq_function(f=100, n_samples=standard_N, **kwargs):
    benchmark = FormulaBenchmark(
        lambda a: math.sin((a + 0.5) * f) / 2,
        {"a": random_feature_sampler},
        N=n_samples,
        train_idx_func=random_train_idx_func,
        task_type="multiclass",
        **kwargs,
    )
    return benchmark


def get_sin_plus_x_function(**kwargs):
    benchmark = FormulaBenchmark(
        lambda a: 0.5 + math.sin(a * 20) / 3 + a / 2.5,
        {"a": single_ordered_feature_sampler},
        N=standard_N,
        train_idx_func=standard_train_idx_func,
        **kwargs,
    )
    return benchmark


def get_linear_plus_heteroscedatic_noise(**kwargs):
    benchmark = FormulaBenchmark(
        lambda a: 0.5 + a + a * (random.random() - 1) * 1.0,
        {"a": single_ordered_feature_sampler},
        N=standard_N,
        train_idx_func=more_train_idx_func,
        **kwargs,
    )
    return benchmark


def get_linear_plus_heteroscedatic_noise_with_gap(**kwargs):
    benchmark = FormulaBenchmark(
        lambda a: 0.5 + (a + 0.7) * random.random() / 4,
        {"a": single_ordered_feature_sampler},
        N=standard_N,
        train_idx_func=more_train_idx_func_with_gap,
        **kwargs,
    )
    return benchmark


def get_linear_plus_homeoscedatic_noise(add_gap=False, trans=0.0, scale=1.0, **kwargs):
    benchmark = FormulaBenchmark(
        lambda a: (0.5 + a + random.random() * 0.2) * scale + trans,
        {"a": single_ordered_feature_sampler},
        N=standard_N,
        train_idx_func=more_train_idx_func_with_gap if add_gap else more_train_idx_func,
        **kwargs,
    )
    return benchmark


def get_generalized_line(n_examples=100, **kwargs):
    benchmark = FormulaBenchmark(
        lambda a: a + random.random() * 0.0,
        {"a": single_ordered_feature_sampler},
        N=n_examples,
        train_idx_func=generalization_idx_func,
        **kwargs,
    )
    return benchmark


def get_generalized_power(p=2, **kwargs):
    offset = 0.5 if p == 2 else 0
    div = 1 if p == 2 else 2
    benchmark = FormulaBenchmark(
        lambda a: math.pow(a, p) * math.pow(2, p) / div - offset,
        {"a": single_ordered_feature_sampler},
        N=standard_N,
        train_idx_func=generalization_idx_func,
        **kwargs,
    )
    return benchmark


def get_generalized_sin_plus_x(hardcore=False, **kwargs):
    benchmark = FormulaBenchmark(
        lambda a: (math.sin(((a + 0.5) * 2 * math.pi) * 2.5) + a) / 3,
        {"a": single_ordered_feature_sampler},
        N=standard_N * 10,
        train_idx_func=one_side_generalization_idx_func
        if hardcore
        else generalization_idx_func,
        **kwargs,
    )
    return benchmark


def get_division(**kwargs):
    benchmark = FormulaBenchmark(
        lambda a, b: a / b,
        {
            "a": lambda N: np.arange(-N // 2, N // 2) / N,
            "b": lambda N: 0.1 + np.random.rand(N),
        },
        N=standard_N,
        train_idx_func=standard_train_idx_func,
        **kwargs,
    )
    return benchmark


def get_multiplication(**kwargs):
    benchmark = FormulaBenchmark(
        lambda a, b: a * b,
        {
            "a": lambda N: np.arange(-N // 2, N // 2) / N,
            "b": lambda N: 0.1 + np.random.rand(N),
        },
        N=standard_N,
        train_idx_func=standard_train_idx_func,
        **kwargs,
    )
    return benchmark


def get_generalized_sin(**kwargs):
    benchmark = FormulaBenchmark(
        lambda a: math.sin(a * 2 * math.pi) / 2,
        {"a": single_ordered_feature_sampler},
        N=standard_N * 10,
        train_idx_func=generalization_idx_func,
        **kwargs,
    )
    return benchmark


class RetrievalBenchmark(SyntheticDataBenchmark):
    def __init__(
        self,
        N=100,
        n_features=2,
        noise=0.01,
        n_prototypes=40,
        discrete_x=False,
        device="cpu",
        task_type="regression",
    ):
        self.N = N
        self.n_features = n_features
        self.discrete_x = discrete_x
        self.y = None
        self.task_type = task_type
        self.device = device

        self.train_idx = np.array([i < 0.8 * N for i in range(N)])
        self.N_PROTOTYPES = n_prototypes

        while self.y is None or len(np.unique(self.y)) == 1:
            data = self.create_dataset()
            self.y = data[:, -1].astype(float)

        if self.task_type == "regression":
            self.y = self.y + np.random.randn(N) * noise
        elif self.task_type == "multiclass":
            self.y = self.y.astype(int)

        # self.y = self.y + np.random.randn(N) * noise
        self.x = (
            data[:, :-1].astype(float) + np.random.randn(N, data.shape[-1] - 1) * noise
        )
        super().__init__(task_type=self.task_type)

    def create_dataset(self):
        ds = []
        prototypes = [
            [random.random() for i in range(0, self.n_features + 1)]
            for j in range(self.N_PROTOTYPES)
        ]
        for sample in range(0, self.N):
            v = copy.copy(prototypes[sample % self.N_PROTOTYPES])
            v[-1] = v[-1] * 5
            ds += [v]
        ds = np.array(ds)
        return ds
