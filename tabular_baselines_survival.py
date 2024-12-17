import torch
import numpy as np
import pandas as pd
from hyperopt import hp
from ..utils import preprocess_and_impute, eval_complete_f, get_scoring_direction
from ...tabular_evaluation_utils import DatasetEvaluation
from tabpfn import datasets

from tabpfn.utils import print_once
from tabpfn.scripts.estimator.scoring_utils import get_score_survival_model

param_grid_hyperopt_survival = {}

"""
Survival
"""

param_grid_hyperopt_survival["lifelines_coxph"] = {
    "alpha": hp.choice("alpha", [0.01, 0.05, 0.1, 0.2]),
    "baseline_estimation_method": hp.choice("baseline_estimation_method", ["breslow"]),
    "penalizer": hp.uniform("penalizer", 0.0, 1.0),
    "l1_ratio": hp.uniform("l1_ratio", 0.0, 1.0),
}


def lifelines_coxph_metric(
    train_ds,
    test_ds,
    random_state=None,
    device=None,
    **kwargs,
):
    # TODO: Hyperparameter tuning
    assert (
        type(train_ds).__name__ == datasets.SurvivalDataset.__name__
    ), "Dataset must be a Survival Dataset"

    from lifelines.fitters.coxph_fitter import CoxPHFitter, ProportionalHazardMixin
    from lifelines.exceptions import ConvergenceError

    class CoxPHFitterWrapper(CoxPHFitter):
        def fit(self, *args, **kwargs):
            try:
                self.failed = False
                return super().fit(*args, **kwargs)
            except ConvergenceError:
                self.failed = True
                return self

        def predict_partial_hazard(self, X, **kwargs):
            if self.failed:
                return np.ones(X.shape[0])
            return -self._model.predict_partial_hazard(X, **kwargs)

    if random_state is not None:
        print_once(
            f"Setting random seed on coxph is not supported yet, ignoring random_state."
        )

    x, y, test_x, attribute_names, categorical_feats = preprocess_and_impute(
        train_ds.x,
        train_ds.y,
        test_ds.x,
        one_hot=True,
        impute=True,
        standardize=True,
        attribute_names=train_ds.attribute_names,
        cat_features=train_ds.categorical_feats,
        is_classification=False,
    )
    data = torch.cat(
        [
            torch.tensor(x),
            train_ds.event_observed.unsqueeze(-1),
            train_ds.y.unsqueeze(-1),
        ],
        -1,
    )
    train_df = pd.DataFrame(data=data.numpy(), columns=attribute_names + ["E", "T"])

    cph = CoxPHFitterWrapper()
    cph.fit(train_df, duration_col="T", event_col="E")
    pred = cph.predict_partial_hazard(test_x)

    return DatasetEvaluation(y=None, pred=pred, additional_args={})


def get_sksurv_metric(
    clf,
    name,
    onehot_drop=None,
    impute=True,
    standardize=True,
    one_hot=True,
    no_tune=None,
    **kwargs,
):
    def sksurv_metric(
        train_ds,
        test_ds,
        metric_used,
        max_time=300,
        device=None,
        no_tune=None,
        random_state=0,
    ):
        assert (
            type(train_ds).__name__ == datasets.SurvivalDataset.__name__
        ), "Dataset must be a Survival Dataset"

        (
            x,
            y,
            test_x,
            attribute_names,
            categorical_feats,
        ) = preprocess_and_impute(
            train_ds.x,
            train_ds.y,
            test_ds.x,
            one_hot=one_hot,
            impute=impute,
            standardize=standardize,
            attribute_names=train_ds.attribute_names,
            cat_features=train_ds.categorical_feats,
            is_classification=False,
            onehot_drop=onehot_drop,
        )

        y_full = np.array(
            list(zip(train_ds.event_observed, y)),
            dtype=[
                ("b", "bool"),
                ("a", "float"),
            ],
        )

        def clf_(**params):
            return clf(**params)

        scorer = get_score_survival_model(metric_used)

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds_eval = eval_complete_f(
                x,
                y_full,
                test_x,
                name,
                param_grid_hyperopt_survival[name],
                clf_,
                scorer,
                max_time,
                no_tune,
                random_state,
                use_metric_as_scorer=True,
            )
        ds_eval.pred = -ds_eval.pred
        return ds_eval

    return sksurv_metric


param_grid_hyperopt_survival["kaplan_meier_estimator"] = {
    "time_min": hp.choice("time_min", [None, hp.uniform("time_min_value", 0, 100)]),
    "reverse": hp.choice("reverse", [True, False]),
    "conf_level": hp.uniform("conf_level", 0.80, 0.99),
    "conf_type": hp.choice("conf_type", [None, "log-log"]),
}


def survival_kaplan_meier_metric(*args, **kwargs):
    from sksurv.nonparametric import kaplan_meier_estimator

    return get_sksurv_metric(clf=kaplan_meier_estimator, name="kaplan_meier_estimator")(
        *args, **kwargs
    )


param_grid_hyperopt_survival["survival_svm"] = {
    "alpha": hp.choice("alpha", [0.1, 1, 10, 100]),
    "rank_ratio": hp.choice("gamma", [0, 0.1, 0.2, 0.5, 0.8, 1]),
}


def survival_svm_metric(*args, **kwargs):
    from sksurv.svm import FastSurvivalSVM

    return get_sksurv_metric(
        clf=FastSurvivalSVM,
        name="survival_svm",
        impute=True,
        standardize=True,
        one_hot=True,
        onehot_drop="First",
    )(*args, **kwargs)


param_grid_hyperopt_survival["survival_grad_boost"] = {
    "loss": hp.choice("loss", ["coxph", "squared"]),
    "learning_rate": hp.loguniform("learning_rate", -3.5, 0.2),
    "n_estimators": hp.choice("n_estimators", range(50, 301, 50)),
    "criterion": hp.choice("criterion", ["friedman_mse", "squared_error"]),
    "min_samples_split": hp.choice("min_samples_split", [1, 2, 4, 6, 10]),
    "min_samples_leaf": hp.choice("min_samples_leaf", [1, 2, 3, 5]),
    "min_weight_fraction_leaf": hp.choice(
        "min_weight_fraction_leaf", [0.0, 0.1, 0.2, 0.5]
    ),
    "max_depth": hp.choice("max_depth", range(1, 6)),
    "subsample": hp.choice("subsample", [0.5, 0.8, 1.0]),
    "dropout_rate": hp.choice("dropout_rate", [0.0, 0.1, 0.2, 0.5]),
    "ccp_alpha": hp.choice("ccp_alpha", [0.0, 0.1, 0.2, 0.5]),
}


def survival_grad_boost_metric(*args, **kwargs):
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis

    return get_sksurv_metric(
        clf=GradientBoostingSurvivalAnalysis, name="survival_grad_boost"
    )(*args, **kwargs)


param_grid_hyperopt_survival["survival_random_forest"] = {
    "n_estimators": hp.choice("n_estimators", range(50, 201, 50)),
    "max_depth": hp.choice("max_depth", [None, 2, 4, 6, 8, 10]),
    "min_samples_split": hp.choice("min_samples_split", [2, 4, 6, 8, 12, 18]),
    "min_samples_leaf": hp.choice("min_samples_leaf", [1, 2, 3, 5, 8, 13]),
    "min_weight_fraction_leaf": hp.choice(
        "min_weight_fraction_leaf", [0.0, 0.1, 0.2, 0.5]
    ),
    "max_features": hp.choice("max_features", ["sqrt", None, "log2"]),
    "bootstrap": hp.choice("bootstrap", [True, False]),
}


def survival_random_forest_metric(*args, **kwargs):
    from sksurv.ensemble import RandomSurvivalForest

    return get_sksurv_metric(
        clf=RandomSurvivalForest,
        name="survival_random_forest",
        one_hot=False,
        impute=True,  # Does not acceot nans
        standardize=False,
    )(*args, **kwargs)


param_grid_hyperopt_survival["survival_tree"] = {
    "max_depth": hp.choice("max_depth", [None] + list(range(3, 15))),
    "min_samples_split": hp.choice("min_samples_split", range(2, 6)),
    "min_samples_leaf": hp.choice("min_samples_leaf", range(1, 10)),
    "min_weight_fraction_leaf": hp.choice(
        "min_weight_fraction_leaf", [0.0, 0.1, 0.2, 0.5]
    ),
    "max_features": hp.choice("max_features", [None, "sqrt", "log2"]),
    "max_leaf_nodes": hp.choice("max_leaf_nodes", [None] + list(range(2, 15))),
}


def survival_tree_metric(*args, **kwargs):
    from sksurv.tree import SurvivalTree

    return get_sksurv_metric(
        clf=SurvivalTree, name="survival_tree", one_hot=False, impute=True
    )(*args, **kwargs)


param_grid_hyperopt_survival["coxph"] = {
    "alpha": hp.choice("alpha", [0, 0.1, 0.5, 1]),
    "ties": hp.choice("ties", ["breslow", "efron"]),
    "n_iter": hp.choice("n_iter", range(50, 201, 50)),
}


def survival_coxph_metric(*args, **kwargs):
    from sksurv.linear_model import CoxPHSurvivalAnalysis

    class CoxPHSurvivalAnalysisWrapper(CoxPHSurvivalAnalysis):
        def fit(self, X, y):
            try:
                return super().fit(X, y)
            except np.linalg.LinAlgError as e:
                if "Matrix is singular" in str(e):
                    self.coef_ = np.zeros(X.shape[1])
                    self.intercept_ = 0.0
                    return self
                else:
                    raise e

        def predict(self, X):
            return super().predict(X)

    try:
        model = get_sksurv_metric(
            clf=CoxPHSurvivalAnalysisWrapper,
            name="coxph",
            one_hot=True,
            impute=True,
            standardize=True,
        )(*args, **kwargs)
        return model
    except ValueError as e:
        if "search direction contains NaN or infinite values" in str(e):
            pred = np.ones((kwargs["test_ds"].y.shape[0]))
            return DatasetEvaluation(
                y=None,
                pred=pred,
                additional_args={},
                algorithm_name="coxph",
            )
        else:
            raise e


param_grid_hyperopt_survival["deephit"] = {
    "num_nodes": hp.choice("num_nodes_shared", [[32, 32], [64, 64], [128, 128]]),
    "batch_norm": hp.choice("batch_norm", [True, False]),
    "dropout": hp.choice("dropout", [0.0, 0.1, 0.5]),
    "batch_size": hp.choice("batch_size", [32, 64, 128]),
    "epochs": hp.choice("epochs", [100, 200, 300, 400, 500]),
    "alpha": hp.choice("alpha", [0.2, 0.4, 0.6]),
    "sigma": hp.choice("sigma", [0.1, 0.2, 0.3]),
    "lr": hp.loguniform("lr", -6, -2),
}


def survival_deephit_pycox_metric(*args, **kwargs):
    from pycox.models import DeepHitSingle
    from pycox.preprocessing.label_transforms import LabTransDiscreteTime
    import torchtuples as tt

    class SklearnWrapperPyCoxDeepHit:
        def __init__(
            self,
            num_nodes=[32, 32],
            batch_norm=True,
            dropout=0.1,
            batch_size=32,
            lr=0.01,
            epochs=500,
            alpha=0.2,
            sigma=0.1,
        ):
            self.num_nodes = num_nodes
            self.batch_norm = batch_norm
            self.dropout = dropout
            self.batch_size = batch_size
            self.lr = lr
            self.epochs = epochs
            self.alpha = alpha
            self.sigma = sigma

        def fit(self, x, y, **kwargs):
            y, event_observed = list(zip(*y.tolist()))[1], list(zip(*y.tolist()))[0]
            from sklearn.model_selection import train_test_split

            x_train, x_val, y_train, y_val, event_train, event_val = train_test_split(
                x, y, event_observed, test_size=0.2, random_state=0
            )

            x_train = torch.tensor(x_train).float()
            x_val = torch.tensor(x_val).float()

            y_train = np.array(y_train)
            y_val = np.array(y_val)

            event_train, event_val = np.array(event_train), np.array(event_val)

            y_train = (y_train, event_train)
            y_val = (torch.tensor(y_val).float(), torch.tensor(event_val).float())
            val = (x_val, y_val)

            class LabTransform(LabTransDiscreteTime):
                def transform(self, durations, events):
                    durations, is_event = super().transform(durations, events > 0)
                    events[is_event == 0] = 0
                    return durations, events.astype("int64")

            num_durations = 10
            labtrans = LabTransform(num_durations)
            y_train = labtrans.fit_transform(*y_train)

            net = tt.practical.MLPVanilla(
                x_train.shape[1],
                self.num_nodes,
                out_features=1,
                batch_norm=self.batch_norm,
                dropout=self.dropout,
            )

            optimizer = tt.optim.AdamWR(
                lr=self.lr, decoupled_weight_decay=0.01, cycle_eta_multiplier=0.8
            )
            self.model = DeepHitSingle(
                net,
                optimizer,
                alpha=self.alpha,
                sigma=self.sigma,
                duration_index=labtrans.cuts,
            )

            callbacks = [tt.callbacks.EarlyStopping()]
            verbose = True

            self.model.fit(
                x_train,
                torch.tensor(y_train).float(),
                self.batch_size,
                self.epochs,
                callbacks,
                verbose,
                val_data=val,
            )

        def predict(self, x):
            x = torch.tensor(x).float()
            return self.model.predict_surv_df(x).values.mean(0)

    return get_sksurv_metric(
        clf=SklearnWrapperPyCoxDeepHit,
        name="deephit",
        standardize=True,
        impute=True,
        one_hot=True,
    )(
        *args,
        **kwargs,
    )


param_grid_hyperopt_survival["coxph_pycox"] = {
    "num_nodes": hp.choice(
        "num_nodes", [[32, 32, 32], [32, 32], [64, 64], [128], [256]]
    ),
    "batch_norm": hp.choice("batch_norm", [True, False]),
    "dropout": hp.choice("dropout", [0.0, 0.1, 0.5]),
    "batch_size": hp.choice("batch_size", [32, 64, 128]),
    "epochs": hp.choice("epochs", [100, 200, 300, 400, 500]),
}


def survival_coxph_pycox_metric(*args, **kwargs):
    # AKA DeepSurv
    from pycox.models import CoxPH
    import torchtuples as tt

    class SklearnWrapperPyCoxCoxPH:
        def __init__(
            self,
            num_nodes=100,
            batch_norm=True,
            dropout=0.1,
            batch_size=32,
            lr=None,
            epochs=500,
        ):
            self.num_nodes = num_nodes
            self.batch_norm = batch_norm
            self.dropout = dropout
            self.batch_size = batch_size
            self.lr = lr
            self.epochs = epochs

        def fit(self, x, y, **kwargs):
            y, event_observed = list(zip(*y.tolist()))[1], list(zip(*y.tolist()))[0]

            from sklearn.model_selection import train_test_split

            x_train, x_val, y_train, y_val, event_train, event_val = train_test_split(
                x, y, event_observed, test_size=0.2, random_state=0
            )

            x_train = torch.tensor(x_train).float()
            x_val = torch.tensor(x_val).float()

            y_train = torch.tensor(y_train).float()
            y_val = torch.tensor(y_val).float()

            if not torch.is_tensor(event_train):
                event_train = torch.tensor(event_train).float()
                event_val = torch.tensor(event_val).float()

            y_train = (y_train, event_train)
            y_val = (y_val, event_val)
            val = (x_val, y_val)

            self.net = tt.practical.MLPVanilla(
                x_train.shape[1],
                self.num_nodes,
                out_features=1,
                batch_norm=self.batch_norm,
                dropout=self.dropout,
            )

            self.model = CoxPH(self.net, tt.optim.Adam)

            if self.lr == None:
                lrfinder = self.model.lr_finder(
                    x_train, y_train, self.batch_size, tolerance=10
                )
                lr = lrfinder.get_best_lr()
                print(f"Found learning rate: {lr}")
                _ = lrfinder.plot()
            else:
                lr = self.lr

            self.model.optimizer.set_lr(lr)

            callbacks = [
                tt.callbacks.EarlyStopping(patience=10),
                tt.callbacks.MonitorMetrics(),
            ]
            verbose = True
            log = self.model.fit(
                x_train,
                y_train,
                self.batch_size,
                self.epochs,
                callbacks,
                verbose,
                val_data=val,
                val_batch_size=self.batch_size,
            )

            _ = log.plot()

            _ = self.model.compute_baseline_hazards()

        def predict(self, x):
            x = torch.tensor(x).float()

            # TODO: Predict survival function
            return self.model.predict_surv_df(x).values.mean(0)

    return get_sksurv_metric(
        clf=SklearnWrapperPyCoxCoxPH,
        name="coxph_pycox",
        standardize=True,
        impute=True,
        one_hot=True,
    )(
        *args,
        **kwargs,
    )


param_grid_hyperopt_survival["ipc_ridge"] = {
    "alpha": hp.loguniform("alpha", -3, 1),
    "fit_intercept": hp.choice("fit_intercept", [True, False]),
    "max_iter": hp.choice("max_iter", [100, 1000, 10000]),
    "positive": hp.choice("positive", [True, False]),
}


def survival_ipc_ridge_metric(*args, **kwargs):
    from sksurv.linear_model import IPCRidge

    class IPCRidgeWrapper(IPCRidge):
        def fit(self, X, y):
            try:
                return super().fit(X, y)
            except:
                self.coef_ = np.zeros(X.shape[1])
                self.intercept_ = 0.0
                return self

        def predict(self, X):
            return -super().predict(X)  # Somehow the sign is flipped

    return get_sksurv_metric(clf=IPCRidgeWrapper, name="ipc_ridge")(*args, **kwargs)


param_grid_hyperopt_survival["coxnet"] = {
    "n_alphas": hp.choice("n_alphas", [50, 100]),
    "alpha_min_ratio": hp.choice(
        "alpha_min_ratio", ["auto", hp.uniform("alpha_min_ratio_value", 0.0001, 0.005)]
    ),
    "l1_ratio": hp.uniform("l1_ratio", 0.0, 1.0),
    "normalize": hp.choice("normalize", [True, False]),
    "max_iter": hp.choice("max_iter", [10000, 50000]),
    "fit_baseline_model": hp.choice("fit_baseline_model", [True]),
}


def survival_coxnet_metric(*args, **kwargs):
    from sksurv.linear_model import CoxnetSurvivalAnalysis

    class CoxnetSurvivalAnalysisWrapper(CoxnetSurvivalAnalysis):
        def __init__(
            self,
            n_alphas=100,
            alphas=None,
            alpha_min_ratio="auto",
            l1_ratio=0.5,
            penalty_factor=None,
            normalize=False,
            copy_X=True,
            tol=1e-7,
            max_iter=100000,
            verbose=False,
            fit_baseline_model=True,  # This needs to be true by default
        ):
            return super().__init__(
                n_alphas=n_alphas,
                alphas=alphas,
                alpha_min_ratio=alpha_min_ratio,
                l1_ratio=l1_ratio,
                penalty_factor=penalty_factor,
                normalize=normalize,
                copy_X=copy_X,
                tol=tol,
                max_iter=max_iter,
                verbose=verbose,
                fit_baseline_model=fit_baseline_model,
            )

    return get_sksurv_metric(
        clf=CoxnetSurvivalAnalysisWrapper,
        name="coxnet",
        onehot_drop="first",
        one_hot=True,
        impute=True,
        standardize=True,
    )(*args, **kwargs)


def survival_auton_metric(
    train_ds,
    test_ds,
    random_state=None,
    **kwargs,
):
    # TODO: Not working yet
    from auton_survival import estimators

    assert (
        type(train_ds).__name__ == datasets.SurvivalDataset.__name__
    ), "Dataset must be a Survival Dataset"

    if random_state is not None:
        print_once(
            f"Setting random seed on auton is not supported yet, ignoring random_state."
        )

    # TODO: Use official preprocessing

    x, y, test_x, attribute_names, categorical_feats = preprocess_and_impute(
        train_ds.x,
        train_ds.y,
        test_ds.x,
        one_hot=True,
        impute=True,
        standardize=True,
        attribute_names=train_ds.attribute_names,
        cat_features=train_ds.categorical_feats,
        is_classification=False,
    )

    from auton_survival.models.dsm import DeepSurvivalMachines

    model = DeepSurvivalMachines()
    model.fit(x, y, train_ds.event_observed.numpy())

    # Predict risk at time horizons.
    predictions = -model.predict_risk(
        test_x.astype(float), t=test_ds.get_time_horizons()
    )
    predictions = np.mean(predictions, axis=1)

    return DatasetEvaluation(
        y=None, pred=predictions, additional_args={"censoring": test_ds.event_observed}
    )


def survival_autoprognosis_metric(
    train_ds,
    test_ds,
    max_time=300,
    random_state=None,
    **kwargs,
):
    # TODO: Not working yet
    # Look at colab: https://colab.research.google.com/drive/1DtZCqebhaYdKB3ci5dr3hT0KvZPaTUOi?usp=sharing#scrollTo=529fd74d
    from autoprognosis.plugins.prediction.classifiers import Classifiers
    from autoprognosis.studies.risk_estimation import RiskEstimationStudy

    assert (
        type(train_ds).__name__ == datasets.SurvivalDataset.__name__
    ), "Dataset must be a Survival Dataset"

    if random_state is not None:
        print_once(
            f"Setting random seed on auton is not supported yet, ignoring random_state."
        )

    x, y, test_x, attribute_names, categorical_feats = preprocess_and_impute(
        train_ds.x,
        train_ds.y,
        test_ds.x,
        one_hot=False,
        impute=False,
        standardize=False,
        attribute_names=train_ds.attribute_names,
        cat_features=train_ds.categorical_feats,
        is_classification=False,
    )

    data = torch.cat(
        [
            torch.tensor(x),
            train_ds.event_observed.unsqueeze(-1),
            torch.tensor(y).unsqueeze(-1),
        ],
        -1,
    )
    train_df = pd.DataFrame(data=data.numpy(), columns=attribute_names + ["E", "T"])
    train_df["E"] = train_df["E"].astype(int)

    study = RiskEstimationStudy(
        study_name="test",
        dataset=train_df,  # pandas DataFrame
        target="E",  # the label column in the dataset
        time_to_event="T",  # the event column in the dataset
        time_horizons=train_ds.get_time_horizons(),
        timeout=max_time,
        num_iter=1,
        num_study_iter=2,
        n_folds_cv=2,
        ensemble_size=1,
        risk_estimators=[
            "survival_xgboost",
        ],  # DELETE THIS LINE FOR BETTER RESULTS.
    )
    model = study.fit()

    # Predict the probabilities of each class using the model
    predictions = model.predict(test_x, test_ds.get_time_horizons())
    predictions = np.mean(predictions.data, axis=1)

    return DatasetEvaluation(y=None, pred=predictions)
