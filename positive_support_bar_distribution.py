import torch
from torch import nn

from .bar_distribution import FullSupportBarDistribution


class PositiveSupportBarDistributionForSurvival(FullSupportBarDistribution):
    """
    This loss function is made to predict survival times, it expects positive survival times.
    Survival times can be censored, i.e. we do not know the exact survival time, but we know that it is greater than
    a certain value.

    Thus the loss consists of two losses, the loss for censored and uncensored samples. Predictions
    that are smaller than the censoring time use the standard log loss. If the survival time is greater than the
    censoring time, the loss will the binary cross entropy weighted for the probability of survival times greater than
    the censoring time.
    """

    def __init__(self, borders, **kwargs):
        super().__init__(borders, **kwargs)
        self.BCE_loss = nn.BCEWithLogitsLoss(reduction="none")

    def assert_support(self, allow_zero_bucket_left=True):
        super().assert_support(allow_zero_bucket_left=True)

    def loss_right_of(self, logits, y):
        """
        Returns the log loss for predicting a value in logits where the corresponding buckets map to a value that
        should be greater than y. I.e. trains for predictions that are greater than y.

        :param logits:
        :param y:
        :return:
        """
        assert self.num_bars > 1

        y = y.view(*logits.shape[:-1])  # no trailing one dimension
        ignore_loss_mask = self.ignore_init(y)  # alters y
        target_sample = self.map_to_bucket_idx(y)  # shape: T x B (same as y)
        target_sample.clamp_(0, self.num_bars - 1)

        logits = torch.log_softmax(logits, -1)

        # Append a zero probability bucket for cumsum
        log_probs = torch.cat(
            [
                logits,
                torch.tensor(float("-inf"), device=logits.device).repeat(
                    logits.shape[0], logits.shape[1], 1
                ),
            ],
            -1,
        )
        # Accumulate the probabilities that lie above the bucket value
        log_probs_right_of = torch.flip(
            torch.logcumsumexp(torch.flip(log_probs, [-1]), -1), [-1]
        )

        # Gather accumulated probabilities higher than the true values
        log_probs_right_of_target = log_probs_right_of.gather(
            -1, target_sample.unsqueeze(-1) + 1
        ).squeeze(-1)

        log_probs_within_bucket = (
            logits[: len(target_sample)]
            .gather(-1, target_sample.unsqueeze(-1))
            .squeeze(-1)
        )

        # Add probability of half buckets
        side_normals = (
            self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
            self.halfnormal_with_p_weight_before(self.bucket_widths[-1]),
        )

        # Calculate the probability within the bucket for lower and upper buckets
        lower_bucket_probability = torch.log(
            side_normals[0].cdf(
                (self.borders[1] - y[target_sample == 0]).clamp(min=0.00000001)
            )
        )
        upper_bucket_probability = torch.log(
            1
            - side_normals[1].cdf(
                (y[target_sample == self.num_bars - 1] - self.borders[-2]).clamp(
                    min=0.00000001
                )
            )
        )

        # Set the lower and upper bucket probabilities
        log_probs_within_bucket[target_sample == 0] += lower_bucket_probability
        log_probs_within_bucket[
            target_sample == self.num_bars - 1
        ] += upper_bucket_probability

        middle_bucket_samples = torch.logical_and(
            target_sample > 0, target_sample < self.num_bars - 1
        )
        # Find the fraction of the bucket that lies to the right of the prediction, falls within [0, 1]
        right_fraction = 1 - (
            (y - self.borders[target_sample]).clamp(min=0)
            / self.bucket_widths[target_sample]
        )  # Tested: Correct
        # Multiply the probability within the bucket by the fraction of the bucket that lies to the right of the prediction
        log_probs_within_bucket[middle_bucket_samples] += torch.log(
            right_fraction[middle_bucket_samples]
        )

        # Add the log probability within the bucket to log_probs_right_of_target
        log_probs_right_of_target = torch.logaddexp(
            log_probs_right_of_target, log_probs_within_bucket
        )

        # Create dummy target value, where class 1 is the correct class
        targets = torch.tensor(1.0, device=log_probs.device).repeat(
            logits.shape[0], logits.shape[1]
        )

        # Compute BCELoss using log probabilities
        loss = self.BCE_loss(log_probs_right_of_target, targets)
        loss[ignore_loss_mask] = 0.0

        # Shape S x B
        return -log_probs_right_of_target

    def forward(self, logits, time, event):
        assert event.shape[2] == 1, "Multi output not yet supported"
        # For all where there was an event: Normal CE loss
        # For all where was no event: Maximize probability right of time
        # logits_with_event, times_with_event = logits[event[:, :, 0]], time[event[:, :, 0]]
        # logits_without_event, times_without_event = logits[~event[:, :, 0]], time[~event[:, :, 0]]
        try:
            from scipy import stats

            print(
                "Censor_fraction",
                float(event[:, :, :].float().detach().mean().cpu().numpy()),
                "Censoring loss",
                float(
                    self.loss_right_of(logits, time)[event[:, :, 0] == False]
                    .mean()
                    .detach()
                    .cpu()
                    .numpy()
                ),
                "CE Loss",
                float(
                    super()
                    .forward(logits, time)[event[:, :, 0] == True]
                    .mean()
                    .cpu()
                    .detach()
                    .numpy()
                ),
                "Mean Pred",
                float(super().mean(logits).cpu().detach().numpy().mean()),
                "Corr Pred",
                stats.spearmanr(
                    super().mean(logits)[event[:, :, 0] == True].cpu().detach().numpy(),
                    time[event[:, :, 0] == True].cpu().detach().numpy().squeeze(-1),
                )[0].mean(),
            )
        except Exception as e:
            print(e)
            pass

        loss_censored_samples = torch.where(
            event[:, :, 0] == True,
            torch.zeros_like(event[:, :, 0]).float(),  # True
            self.loss_right_of(logits, time),
        )
        loss_observed_samples = torch.where(
            event[:, :, 0] == True,
            super().forward(logits, time),  # True
            torch.zeros_like(event[:, :, 0]).float(),
        )

        return (
            loss_censored_samples + loss_observed_samples,
            loss_censored_samples,
            loss_observed_samples,
        )
