import warnings
import math

import numpy as np

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

from ...tabular_metrics import get_scoring_direction
from ...tabular_metrics import get_task_type

from ..utils import (
    preprocess_and_impute,
    eval_complete_f,
    MULTITHREAD,
)

from hyperopt import hp

# XGBoost
# Hyperparameter space: https://arxiv.org/pdf/2106.03253.pdf
param_grid_hyperopt = {
    "learning_rate": hp.loguniform("learning_rate", -7, math.log(1)),
    "max_depth": hp.randint("max_depth", 1, 10),
    "subsample": hp.uniform("subsample", 0.2, 1),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.2, 1),
    "colsample_bylevel": hp.uniform("colsample_bylevel", 0.2, 1),
    "min_child_weight": hp.loguniform("min_child_weight", -16, 5),
    "alpha": hp.loguniform("alpha", -16, 2),
    "lambda": hp.loguniform("lambda", -16, 2),
    "gamma": hp.loguniform("gamma", -16, 2),
    "n_estimators": hp.randint(
        "n_estimators", 100, 4000
    ),  # This is smaller than in paper
}


def get_score_xgb_survival_model(metric_used):
    def score_survival_model(model, X, y):
        censoring, y = y > 0, np.abs(y.astype(float))
        prediction = model.predict(X)

        if np.array(censoring).mean() == 0:
            return 0.5

        result = metric_used(target=y, pred=-prediction, event_observed=censoring)
        result *= get_scoring_direction(metric_used)

        return result

    return score_survival_model


def survival_xgb_metric(
    train_ds,
    test_ds,
    metric_used,
    max_time=300,
    no_tune=None,
    gpu_id=None,
    random_state=0,
    **kwargs,
):
    import xgboost as xgb

    # XGB Documentation:
    # XGB handles categorical data appropriately without using One Hot Encoding, categorical features are experimetal
    # XGB handles missing values appropriately without imputation

    if gpu_id is not None:
        gpu_params = {"tree_method": "gpu_hist", "gpu_id": gpu_id}
    else:
        gpu_params = {}

    y, test_y = train_ds.y, test_ds.y
    is_survival = get_task_type(metric_used) == "survival"
    assert is_survival, "This method only works for survival tasks"

    attribute_names = train_ds.attribute_names

    event_observed = train_ds.event_observed.clone()
    y = y * event_observed.apply_(lambda x: 1 if x else -1).numpy()

    x, y, test_x, _, _ = preprocess_and_impute(
        train_ds.x,
        y,
        test_ds.x,
        one_hot=False,
        impute=False,
        standardize=False,
        attribute_names=attribute_names,
        cat_features=train_ds.categorical_feats,
        is_classification=get_task_type(metric_used) == "multiclass",
        return_pandas=True,
    )

    # XGB expects categorical features to be of type category
    test_x.loc[:, test_x.dtypes == "int"] = test_x.loc[
        :, test_x.dtypes == "int"
    ].astype("category")
    x.loc[:, x.dtypes == "int"] = x.loc[:, x.dtypes == "int"].astype("category")

    def clf_(**params):
        return xgb.XGBRegressor(
            use_label_encoder=False,
            nthread=MULTITHREAD,
            objective="survival:cox",
            enable_categorical=True,
            **params,
            **gpu_params,
        )

    scorer = get_score_xgb_survival_model(metric_used)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds_eval = eval_complete_f(
            x,
            y,
            test_x,
            "xgb_survival",
            param_grid_hyperopt,
            clf_,
            scorer,
            max_time,
            no_tune,
            random_state,
            method_name="xgb_survival",
            use_metric_as_scorer=True,
            verbose=True,
        )

    ds_eval.pred = -ds_eval.pred

    return ds_eval
