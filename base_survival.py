from __future__ import annotations

import numpy as np
import torch

from typing import Optional, Tuple, Any, Dict, List, Literal

from .configs import PreprocessorConfig
from .base import TabPFNRegressor
from ...datasets.survival_datasets import SurvivalDataset
from tabpfn.utils import normalize_data
from tabpfn.scripts.estimator.scoring_utils import score_survival

SurvivalOptimizationMetricType = Literal[
    "cindex",
    "mean",
    "median",
    "mode",
    "risk_score",
    "risk_score_weighted",
    "risk_score_weighted_2",
    "risk_score_weighted_3",
    "risk_score_capped",
    None,
]


class SurvivalRegressorMixin:
    def fit_separate_censoring(self, X, event_times, censoring):
        X = X
        y = event_times

        assert (
            y.min() >= 0
        ), f"Invalid y values, all ys must be greater than 0. You can simply offset y by {y.min()}."

        if len(censoring.shape) == 1:
            censoring = censoring.unsqueeze(1)
        if len(censoring.shape) == 2:
            censoring = censoring.unsqueeze(1)
        assert (
            censoring.shape[1] == 1
            and censoring.shape[2] == 1
            and censoring.shape[0] == X.shape[0]
        ), f"Invalid censoring shape {censoring.shape}"
        assert (
            censoring.min() >= 0 and censoring.max() <= 1
        ), f"Invalid censoring values {censoring.min()} {censoring.max()} {censoring}"

        return self.fit(X, y, additional_y={"event": censoring})


class TabPFNSurvivalRegressor(TabPFNRegressor, SurvivalRegressorMixin):
    metric_type = SurvivalOptimizationMetricType
    task_type = "survival"

    def __init__(
        self,
        model: Optional[Any] = None,
        device: str = "cpu",
        model_path: str = "",
        batch_size_inference: int = None,
        fp16_inference: bool = False,
        model_config: Optional[Dict] = None,
        n_estimators: int = 10,
        preprocess_transforms: Tuple[PreprocessorConfig, ...] = (
            PreprocessorConfig("none"),
            PreprocessorConfig("power", categorical_name="numeric"),
        ),
        feature_shift_decoder: str = "shuffle",
        normalize_with_test: bool = False,
        average_logits: bool = True,
        optimize_metric: SurvivalOptimizationMetricType = "cindex",
        seed: Optional[int] = 0,
        transformer_predict_kwargs: Optional[Dict] = None,
        show_progress: bool = True,
        save_peak_memory: Literal["True", "False", "auto"] = "True",
        softmax_temperature: Optional[float] = 0.0,
        use_poly_features=False,
        cancel_nan_borders: bool = True,
        super_bar_dist_averaging: bool = False,
        max_poly_features=50,
        transductive=False,
        remove_outliers=0.0,
        regression_y_preprocess_transforms: Optional[Tuple[Optional[str], ...]] = (
            None,
            "power",
        ),
        add_fingerprint_features: bool = False,
        subsample_samples: float = -1,
        maximum_free_memory_in_gb: Optional[float] = None,
        split_test_samples: float | str = 1,
    ):
        """
        According to Sklearn API we need to pass all parameters to the super class constructor without **kwargs or *args
        """
        super().__init__(
            model=model,
            device=device,
            model_path=model_path,
            batch_size_inference=batch_size_inference,
            fp16_inference=fp16_inference,
            model_config=model_config,
            n_estimators=n_estimators,
            preprocess_transforms=preprocess_transforms,
            feature_shift_decoder=feature_shift_decoder,
            normalize_with_test=normalize_with_test,
            average_logits=average_logits,
            optimize_metric=optimize_metric,
            cancel_nan_borders=cancel_nan_borders,
            super_bar_dist_averaging=super_bar_dist_averaging,
            seed=seed,
            transformer_predict_kwargs=transformer_predict_kwargs,
            show_progress=show_progress,
            save_peak_memory=save_peak_memory,
            softmax_temperature=softmax_temperature,
            use_poly_features=use_poly_features,
            max_poly_features=max_poly_features,
            transductive=transductive,
            remove_outliers=remove_outliers,
            regression_y_preprocess_transforms=regression_y_preprocess_transforms,
            add_fingerprint_features=add_fingerprint_features,
            subsample_samples=subsample_samples,
            maximum_free_memory_in_gb=maximum_free_memory_in_gb,
            split_test_samples=split_test_samples,
        )

    def init_model_and_get_model_config(self):
        super().init_model_and_get_model_config()
        self.normalize_with_min_ = self.c_processed_.get(
            "survival_normalize_with_min", True
        )

    def fit(self, X, y, additional_y=None):
        if (
            type(y) == np.ndarray
            and y.dtype.names is not None
            and len(y.dtype.names) > 1
        ):
            unzipped = list(zip(*y))
            return self.fit_separate_censoring(
                X,
                event_times=torch.tensor(unzipped[1]),
                censoring=torch.tensor(unzipped[0]),
            )

        return super().fit(X, y, additional_y=additional_y)

    def cluster_survival_curves(self, X, n_clusters=3):
        outs = self.predict_full(X)
        cum_sum = np.cumsum(outs["buckets"], axis=1)

        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(
            cum_sum
        )
        preds = kmeans.predict(cum_sum)

        return preds, kmeans

    def plot_survival_curves(self, curves=None, labels=None, X=None, outs=None):
        import matplotlib.pyplot as plt

        if outs is None:
            outs = self.predict_full(X)

        if curves is None:
            curves = np.cumsum(outs["buckets"], axis=1)

        if labels is None:
            labels = np.array([1 for _ in range(curves.shape[0])])

        cum_sum = np.cumsum(outs["buckets"], axis=1)

        right_side = np.argmax(
            [
                (cum_sum[:, i] > 0.98).mean() > 0.95
                for i in range(outs["criterion"].borders.shape[-1] - 1)
            ]
        )

        fig, ax = plt.subplots()
        for i in np.unique(labels):
            ax.plot(
                outs["criterion"].borders.cpu().detach().numpy()[1 : right_side + 1],
                curves[labels == i, :right_side].T,
                label=i,
                color=plt.cm.tab20(i),
            )
            # ax.legend()
            ax.title.set_text("Clustered survival curves")

    def get_optimization_mode(self):
        optimize_metric = (
            "cindex" if self.optimize_metric is None else self.optimize_metric
        )
        if optimize_metric in ["cindex", "mean"]:
            return "mean"
        elif optimize_metric in ["median"]:
            return "median"
        elif optimize_metric in ["mode"]:
            return "mode"
        elif optimize_metric in ["risk_score"]:
            return "risk_score"
        elif optimize_metric in ["risk_score_capped"]:
            return "risk_score_capped"
        elif optimize_metric in ["risk_score_weighted"]:
            return "risk_score_weighted"
        elif optimize_metric in ["risk_score_weighted_2"]:
            return "risk_score_weighted_2"
        elif optimize_metric in ["risk_score_weighted_3"]:
            return "risk_score_weighted_3"
        elif optimize_metric in ["risk_score_weighted_4"]:
            return "risk_score_weighted_4"
        else:
            raise ValueError(f"Unknown metric {optimize_metric}")

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)

        opt_metric = (
            self.optimize_metric if self.optimize_metric is not None else "cindex"
        )

        censoring, y = list(zip(*y.tolist()))[0], list(zip(*y.tolist()))[1]

        return score_survival(
            opt_metric, y, y_pred, event_observed=censoring, sample_weight=sample_weight
        )

    def normalize_y_labels_fit(self):
        from tabpfn.model.positive_support_bar_distribution import (
            PositiveSupportBarDistributionForSurvival,
        )

        if self.normalize_with_min_:
            y_full, (data_mean, data_std) = normalize_data(
                self.y_[:, None].float(),
                normalize_positions=len(self.y_),
                std_only=False,
                mean=self.y_[:, None].float().min(),
                return_scaling=True,
                clip=False,
            )

            return y_full, (data_mean, data_std)
        else:
            assert not isinstance(
                self.model_processed_.criterion,
                PositiveSupportBarDistributionForSurvival,
            )
            # return self.y_[:, None].float(), (torch.tensor(0.0), torch.tensor(1.0))
            return super().normalize_y_labels_fit()

    def normalize_y_labels_prediction(self, y_full, eval_position):
        from tabpfn.model.positive_support_bar_distribution import (
            PositiveSupportBarDistributionForSurvival,
        )

        if self.normalize_with_min_:
            y_min = (
                self.y_[:, None].float().min()
                if self.fit_at_predict_time
                else self.data_mean_
            )
            y_full, (data_mean, data_std) = normalize_data(
                y_full,
                normalize_positions=eval_position,
                return_scaling=True,
                mean=y_min,
                std=None if self.fit_at_predict_time else self.data_std_,
                clip=False,
                std_only=False,
            )

            return y_full, (data_mean, data_std)
        else:
            assert not isinstance(
                self.model_processed_.criterion,
                PositiveSupportBarDistributionForSurvival,
            )
            # return self.y_[:, None].float(), (torch.tensor(0.0), torch.tensor(1.0))
            return super().normalize_y_labels_prediction(y_full, eval_position)

    def predict_full(self, X, additional_y=None, get_additional_outputs=None) -> dict:
        assert additional_y is None

        if type(X) == np.ndarray:
            X = torch.tensor(X).float()

        additional_y = {
            "event": (
                SurvivalDataset.get_missing_event_indicator()
                * torch.ones_like(X[:, 0]).unsqueeze(1).unsqueeze(1)
            )
        }

        pred = super().predict_full(
            X,
            additional_y=additional_y,
        )

        logits = torch.tensor(pred["logits"]).clone()
        risk_score = pred["criterion"].cdf_temporary(logits)
        # TODO: Use cdf here

        # Cutoff at last observed time
        last_observed = self.y_.max()
        last_observed_borders = pred["criterion"].borders > last_observed

        risk_score_capped = pred["criterion"].cdf_temporary(
            logits, last_observed_borders=last_observed_borders
        )

        # TODO: Weighted risk score, where we weight the aggregated risks by the probability of any
        #  event happening before that time
        #  In practice this would overweight the early risk positions
        probs = logits.softmax(-1)
        cumprobs = -torch.cumsum(probs, -1)

        idx = self.additional_y_["event"] == 1
        idx = self.y_[idx.flatten()].unsqueeze(-1) > pred["criterion"].borders
        idx = torch.argmin(idx[:-1].long(), -1).repeat(cumprobs.shape[0], 1)
        cumprobs_idx = torch.gather(cumprobs, 1, idx.long())
        risk_score_weighted = cumprobs_idx.mean(-1)

        idx = self.y_.unsqueeze(-1) > pred["criterion"].borders
        idx = torch.argmin(idx[:-1].long(), -1).repeat(cumprobs.shape[0], 1)
        cumprobs_idx = torch.gather(cumprobs, 1, idx.long())
        risk_score_weighted_2 = cumprobs_idx.mean(-1)

        cumcumprobs = torch.cumsum(pred["criterion"].bucket_widths * cumprobs, -1)
        idx = self.y_.unsqueeze(-1) > pred["criterion"].borders
        idx = torch.argmin(idx[:-1].long(), -1).repeat(cumcumprobs.shape[0], 1)
        cumprobs_idx = torch.gather(cumcumprobs, 1, idx.long())
        risk_score_weighted_3 = cumprobs_idx.mean(-1)

        # probs = logits.softmax(-1)
        # cumprobs = -torch.cumsum(probs, -1)

        # idx = self.y_.unsqueeze(-1) < pred["criterion"].borders
        # idx = idx.diff(1, 1)

        # Create gaussian kernels
        # kernel = Variable(torch.FloatTensor([[[0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006]]]))
        # kernel = torch.FloatTensor([[[0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006]]])
        # Apply smoothing
        # idx = idx.transpose(0, 1)
        # idx = torch.nn.functional.conv1d(idx.float(), kernel, padding=3)
        # idx = idx.transpose(0,1)

        # idx = idx[:, 1:]# * pred["criterion"].bucket_widths
        # idx = idx / idx.sum(1).unsqueeze(1)
        # risk_score_weighted_4 = (cumprobs * idx.sum(0))# * pred["criterion"].bucket_widths
        # risk_score_weighted_4 = risk_score_weighted_4.sum(-1)

        pred.update(
            {
                "risk_score": risk_score.numpy(),
                "risk_score_capped": risk_score_capped.numpy(),
                "risk_score_weighted": risk_score_weighted.numpy(),
                "risk_score_weighted_2": risk_score_weighted_2.numpy(),
                "risk_score_weighted_3": risk_score_weighted_3.numpy(),
                # "risk_score_weighted_4": risk_score_weighted_4.numpy(),
            }
        )
        return pred
