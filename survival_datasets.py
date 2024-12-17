from collections import defaultdict
import requests
from pathlib import Path
import os
from . import TabularDataset
import torch

_DATA_OVERRIDE = os.environ.get("PYCOX_DATA_DIR", None)
if _DATA_OVERRIDE:
    _PATH_DATA = Path(_DATA_OVERRIDE)
else:
    _PATH_ROOT = Path("./pycox")  # Path(pycox.__file__).parent
    _PATH_DATA = _PATH_ROOT / "datasets" / "data"
_PATH_DATA.mkdir(parents=True, exist_ok=True)


def get_survival_benchmark(min_samples, filter_for_nan, max_samples, num_feats):
    from SurvSet.data import SurvLoader
    from .survival_datasets import get_lifelines_survival

    loader = SurvLoader()
    ds_df = loader.df_ds
    ds_df = ds_df[ds_df["n"] <= max_samples]
    ds_df = ds_df[ds_df["n_ohe"] + ds_df["n_fac"] + ds_df["n_num"] <= num_feats]
    ds_df = ds_df[ds_df["n"] >= min_samples]

    datasets = [
        get_lifelines_survival(
            name,
            list(loader.load_dataset(ds_name=name).values())[0],
            "time",
            "event",
            shuffled=True,
        )
        for name in ds_df.ds
    ]
    return datasets, None


class SurvivalDataset(TabularDataset):
    def __init__(self, event_observed: torch.tensor, y: torch.tensor, **kwargs):
        """ """
        self.task_type = "survival"
        if "task_type" in kwargs:
            del kwargs["task_type"]
        super().__init__(task_type=self.task_type, y=y, **kwargs)
        self.event_observed = (
            event_observed if event_observed is not None else torch.ones_like(y)
        )
        # self.x = SurvivalDataset.append_censoring_to_x(self.x, event_observed)
        self.attribute_names = self.attribute_names

    @staticmethod
    def get_missing_event_indicator():
        return -1

    def censoring_independent_of_X(self, predictor=None):
        from tabpfn.scripts.tabular_evaluation import evaluate_simple
        import pickle

        # If independence is already saved, return it
        if os.path.exists(
            f"censoring_independence/independence_survival_{self.get_dataset_identifier()}.pkl"
        ):
            with open(
                f"censoring_independence/independence_survival_{self.get_dataset_identifier()}.pkl",
                "rb",
            ) as f:
                return pickle.load(f)

        if predictor is None:
            return -2

        ds = copy.deepcopy(self)

        independent = -1

        # Flip censoring
        ds.event_observed = (~(ds.event_observed.bool())).float()

        try:
            metrics_tabpfn, r_tabpfn = evaluate_simple(
                predictor, [ds], task_type="survival"
            )
            score = r_tabpfn["mean_cindex"]
            if score < 0.52:
                independent = 1

        except Exception as e:
            print(f"Error in independence test: {e}")
            independent = -1

        # Save as pickle independence
        with open(
            f"censoring_independence/independence_survival_{self.get_dataset_identifier()}.pkl",
            "wb",
        ) as f:
            pickle.dump(independent, f)

        return independent

    def has_global_censoring(self) -> bool:
        censored_samples_y = self.y[self.event_observed == 0]
        _, counts = np.unique(censored_samples_y, return_counts=True)
        counts = counts / len(censored_samples_y)

        return (counts > 0.5).any()

    def get_time_horizons(self):
        horizons = np.linspace(0.1, 0.9, 20).tolist()
        times = np.quantile(self.y[self.event_observed == 1], horizons).tolist()
        return times

    def infer_and_set_categoricals(self) -> None:
        super().infer_and_set_categoricals()

    def event(self):
        return self.event_observed
        # return self.x[:, 0:1]

    # @property
    # def x(self):
    #    return torch.cat([self.x_data, self.censoring.unsqueeze(-1)], -1)

    def __repr__(self):
        return f"{self.name}"

    def __getitem__(self, indices):
        ds = super().__getitem__(indices)
        ds.event_observed = self.event_observed[indices]

        return ds


def get_gbsg():
    # !pip install pycox --no-deps

    df = gbsg.read_df()

    return get_lifelines_survival("gbsg", df, "duration", "event")


def get_nwtco():
    # !pip install pycox --no-deps

    df = nwtco.read_df()

    return get_lifelines_survival("nwtco", df, "edrel", "rel")


def get_metabric():
    # !pip install pycox --no-deps

    df = metabric.read_df()

    return get_lifelines_survival("metabric", df, "duration", "event")


def get_waltons():
    from lifelines.datasets import load_waltons

    df = load_waltons()  # returns a Pandas DataFrame
    return get_lifelines_survival("waltons", df, "T", "E")


def get_rossi():
    from lifelines.datasets import load_rossi

    df = load_rossi()  # returns a Pandas DataFrame
    return get_lifelines_survival("rossi", df, "week", "arrest")


def get_dd():
    from lifelines.datasets import load_dd

    df = load_dd()  # returns a Pandas DataFrame
    return get_lifelines_survival("dd", df, "duration", "observed")


def get_lifelines_survival(name, df, t, e, shuffled=True):
    cat_columns = df.select_dtypes(["object", "category"]).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.astype("category"))
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

    df = df.sample(frac=1, random_state=0) if shuffled else df
    T = df[t]
    E = df[e]
    df = df.drop(columns=[t, e])

    ds = SurvivalDataset(
        x=torch.tensor(df.values).float(),
        y=torch.tensor(T).float(),
        name=name,
        attribute_names=list(df.columns),
        event_observed=torch.tensor(E).float(),
        extra_info={"did": name},
    )
    ds.infer_and_set_categoricals()

    ds.n_splits = 5 if ds.x.shape[0] > 200 else 2

    return ds


def get_surv_loader_datasets(exclude=[]):
    from SurvSet.data import SurvLoader

    loader = SurvLoader()
    # List of available datasets and meta - info
    # dfs = loader.df_ds[np.logical_and(loader.df_ds.n < 10000, (~loader.df_ds.is_td))]
    ds_lst = loader.df_ds[~loader.df_ds["is_td"]]["ds"].to_list()
    ds_keep = []

    for i, ds in enumerate(ds_lst):
        anno = loader.df_ds.query("ds == @ds").T.to_dict()
        anno = anno[list(anno)[0]]
        df, ref = loader.load_dataset(ds).values()

        if anno["is_td"]:
            continue
        if df.shape[1] > 200 or df.shape[1] < 10:
            continue
        if df.shape[0] > 5000:
            continue
        for n in exclude:
            if n == anno["ds"]:
                continue
        print(df.shape, anno, ref)
        ds = get_lifelines_survival(anno["ds"], df, "time", "event", shuffled=True)
        ds_keep += [ds]

    return ds_keep


def get_support():
    # Install
    # git clone https://github.com/autonlab/auton-survival.git
    # mv auton-survival/auton_survival/ auton_survival
    # rm auton-survival/ -rf
    from auton_survival import datasets as auton_datasets

    outcomes, features = auton_datasets.load_dataset("SUPPORT")
    features = features.assign(event=outcomes.event, time=outcomes.time)
    return get_lifelines_survival("SUPPORT", features, "time", "event", shuffled=True)


def get_framingham():
    # Install
    # git clone https://github.com/autonlab/auton-survival.git
    # mv auton-survival/auton_survival/ auton_survival
    # rm auton-survival/ -rf
    from .survival import _load_framingham_dataset

    features, times, events = _load_framingham_dataset(sequential=False)
    features = features.assign(event=events, time=times)
    return get_lifelines_survival(
        "FRAMINGHAM", features, "time", "event", shuffled=True
    )


def get_pbc():
    # Install
    # git clone https://github.com/autonlab/auton-survival.git
    # mv auton-survival/auton_survival/ auton_survival
    # rm auton-survival/ -rf

    features, times, events = _load_pbc_dataset(sequential=False)
    features = features.assign(event=events, time=times)
    return get_lifelines_survival("PBC", features, "time", "event", shuffled=True)


def get_flchain():
    # !pip install -U scikit-learn==0.21.3 --no-deps
    from sksurv import datasets as sksurv_datasets

    df = sksurv_datasets.load_flchain()  # returns a Pandas DataFrame
    return get_scikit_survival("flchain", df)


def get_gbsg2():
    from sksurv import datasets as sksurv_datasets

    df = sksurv_datasets.load_gbsg2()  # returns a Pandas DataFrame
    return get_scikit_survival("gbsg2", df)


def get_whas500():
    """
    NAME:
    Worcester Heart Attack Study WHAS500 Data (whas500.dat)

    SIZE:
    500 Observations, 22 variables

    SOURCE:
    Worcester Heart Attack Study data from Dr. Robert J. Goldberg of
    the Department of Cardiology at the University of Massachusetts Medical
    School.

    REFERENCE:
    Hosmer, D.W. and Lemeshow, S. and May, S. (2008)
    Applied Survival Analysis: Regression Modeling of Time to Event Data:
    Second Edition, John Wiley and Sons Inc., New York, NY

    DESCRIPTIVE ABSTRACT:
    The main goal of this study is to describe factors associated
    with trends over time in the incidence and survival rates following
    hospital admission for acute myocardial infarction (MI).  Data have been
    collected during thirteen 1-year periods beginning in 1975 and extending
    through 2001 on all MI patients admitted to hospitals in the Worcester,
    Massachusetts Standard Metropolitan Statistical Area.

    DISCLAIMER:
    This data is also available at the following Wiley's FTP site:
    ftp//ftp.wiley.com/public/sci_tech_med/survival
    """
    from sksurv import datasets as sksurv_datasets

    df = sksurv_datasets.load_whas500()  # returns a Pandas DataFrame
    return get_scikit_survival(
        "whas500",
        df,
        description="Worcester Heart Attack Study. Data have been collected during thirteen 1-year periods beginning in 1975 and extending through 2001 on all MI patients admitted to hospitals in the Worcester, Massachusetts Standard Metropolitan Statistical Area.",
    )


def get_veterans_lung_cancer():
    from sksurv import datasets as sksurv_datasets

    df = sksurv_datasets.load_veterans_lung_cancer()  # returns a Pandas DataFrame
    return get_scikit_survival("veterans_lung_cancer", df)


def get_breast_cancer():
    """
    Load and return the breast cancer dataset

    The dataset has 198 samples and 80 features. The endpoint is the presence of distance metastases, which occurred for 51 patients (25.8%).

    Materials and Methods: Gene expression profiling of frozen samples from 198 N- systemically untreated patients was performed at the Bordet Institute, blinded to clinical data and independent of Veridex. Genomic risk was defined by Veridex, blinded to clinical data.

    https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE7390

    Desmedt, C., Piette, F., Loi et al.: “Strong Time Dependence of the 76-Gene Prognostic Signature for Node-Negative Breast Cancer Patients in the TRANSBIG Multicenter Independent Validation Series.” Clin. Cancer Res. 13(11), 3207–14 (2007)
    """
    from sksurv import datasets as sksurv_datasets

    df = sksurv_datasets.load_breast_cancer()  # returns a Pandas DataFrame
    return get_scikit_survival("breast_cancer", df)


def get_aids():
    from sksurv import datasets as sksurv_datasets

    df = sksurv_datasets.load_aids()  # returns a Pandas DataFrame
    return get_scikit_survival("aids", df)


def get_scikit_survival(name, sksurv_ds, shuffled=True, description=None):
    #!pip install scikit-survival==0.10.0 --no-deps
    event, time = list(zip(*sksurv_ds[1].tolist()))
    sksurv_ds = sksurv_ds[0].assign(event=event, time=time)
    ds = get_lifelines_survival(name, sksurv_ds, "time", "event", shuffled=shuffled)
    ds.description = description
    return ds


class _DatasetLoader:
    """Abstract class for loading data sets."""

    name = NotImplemented
    _checksum = None

    def __init__(self):
        self.path = _PATH_DATA / f"{self.name}.feather"

    def read_df(self):
        if not self.path.exists():
            print(f"Dataset '{self.name}' not locally available. Downloading...")
            self._download()
            print(f"Done")
        df = pd.read_feather(self.path)
        df = self._label_cols_at_end(df)
        return df

    def _download(self):
        raise NotImplementedError

    def delete_local_copy(self):
        if not self.path.exists():
            raise RuntimeError("File does not exists.")
        self.path.unlink()

    def _label_cols_at_end(self, df):
        if hasattr(self, "col_duration") and hasattr(self, "col_event"):
            col_label = [self.col_duration, self.col_event]
            df = df[list(df.columns.drop(col_label)) + col_label]
        return df

    def checksum(self):
        """Checks that the dataset is correct.

        Returns:
            bool -- If the check passed.
        """
        if self._checksum is None:
            raise NotImplementedError("No available comparison for this dataset.")
        df = self.read_df()
        return self._checksum_df(df)

    def _checksum_df(self, df):
        if self._checksum is None:
            raise NotImplementedError("No available comparison for this dataset.")
        import hashlib

        val = get_checksum(df)
        return val == self._checksum


def get_checksum(df):
    import hashlib

    val = hashlib.sha256(df.to_csv().encode()).hexdigest()
    return val


def download_from_rdatasets(package, name):
    datasets = (
        pd.read_csv(
            "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/datasets.csv"
        )
        .loc[lambda x: x["Package"] == package]
        .set_index("Item")
    )
    if not name in datasets.index:
        raise ValueError(f"Dataset {name} not found.")
    info = datasets.loc[name]
    url = info.CSV
    return pd.read_csv(url), info


class _DatasetRdatasetsSurvival(_DatasetLoader):
    """Data sets from Rdataset survival."""

    def _download(self):
        df, info = download_from_rdatasets("survival", self.name)
        self.info = info
        df.to_feather(self.path)


class _Flchain(_DatasetRdatasetsSurvival):
    """Assay of serum free light chain (FLCHAIN).
    Obtained from Rdatasets (https://github.com/vincentarelbundock/Rdatasets).

    A study of the relationship between serum free light chain (FLC) and mortality.
    The original sample contains samples on approximately 2/3 of the residents of Olmsted
    County aged 50 or greater.

    For details see http://vincentarelbundock.github.io/Rdatasets/doc/survival/flchain.html

    Variables:
        age:
            age in years.
        sex:
            F=female, M=male.
        sample.yr:
            the calendar year in which a blood sample was obtained.
        kappa:
            serum free light chain, kappa portion.
        lambda:
            serum free light chain, lambda portion.
        flc.grp:
            the FLC group for the subject, as used in the original analysis.
        creatinine:
            serum creatinine.
        mgus:
            1 if the subject had been diagnosed with monoclonal gammapothy (MGUS).
        futime: (duration)
            days from enrollment until death. Note that there are 3 subjects whose sample
            was obtained on their death date.
        death: (event)
            0=alive at last contact date, 1=dead.
        chapter:
            for those who died, a grouping of their primary cause of death by chapter headings
            of the International Code of Diseases ICD-9.

    """

    name = "flchain"
    col_duration = "futime"
    col_event = "death"
    _checksum = "ec12748a1aa5790457c09793387337bb03b1dc45a22a2d58a8c2b9ad1f2648dd"

    def read_df(self, processed=True):
        """Get dataset.

        If 'processed' is False, return the raw data set.
        See the code for processing.

        Keyword Arguments:
            processed {bool} -- If 'False' get raw data, else get processed (see '??flchain.read_df').
                (default: {True})
        """
        df = super().read_df()
        if processed:
            df = (
                df.drop(["chapter", "Unnamed: 0"], axis=1)
                .loc[lambda x: x["creatinine"].isna() == False]
                .reset_index(drop=True)
                .assign(sex=lambda x: (x["sex"] == "M"))
            )

            categorical = ["sample.yr", "flc.grp"]
            for col in categorical:
                df[col] = df[col].astype("category")
            for col in df.columns.drop(categorical):
                df[col] = df[col].astype("float32")
        return df


class _Nwtco(_DatasetRdatasetsSurvival):
    """Data from the National Wilm's Tumor Study (NWTCO)
    Obtained from Rdatasets (https://github.com/vincentarelbundock/Rdatasets).

    Measurement error example. Tumor histology predicts survival, but prediction is stronger
    with central lab histology than with the local institution determination.

    For details see http://vincentarelbundock.github.io/Rdatasets/doc/survival/nwtco.html

    Variables:
        seqno:
            id number
        instit:
            histology from local institution
        histol:
            histology from central lab
        stage:
            disease stage
        study:
            study
        rel: (event)
            indicator for relapse
        edrel: (duration)
            time to relapse
        age:
            age in months
        in.subcohort:
            included in the subcohort for the example in the paper

    References
        NE Breslow and N Chatterjee (1999), Design and analysis of two-phase studies with binary
        outcome applied to Wilms tumor prognosis. Applied Statistics 48, 457–68.
    """

    name = "nwtco"
    col_duration = "edrel"
    col_event = "rel"
    _checksum = "5aa3de698dadb60154dd59196796e382739ff56dc6cbd39cfc2fda50d69d118e"

    def read_df(self, processed=True):
        """Get dataset.

        If 'processed' is False, return the raw data set.
        See the code for processing.

        Keyword Arguments:
            processed {bool} -- If 'False' get raw data, else get processed (see '??nwtco.read_df').
                (default: {True})
        """
        df = super().read_df()
        if processed:
            df = df.assign(
                instit_2=df["instit"] - 1,
                histol_2=df["histol"] - 1,
                study_4=df["study"] - 3,
                stage=df["stage"].astype("category"),
            ).drop(["Unnamed: 0", "seqno", "instit", "histol", "study"], axis=1)
            for col in df.columns.drop("stage"):
                df[col] = df[col].astype("float32")
            df = self._label_cols_at_end(df)
        return df


class _DatasetDeepSurv(_DatasetLoader):
    _dataset_url = "https://raw.githubusercontent.com/jaredleekatzman/DeepSurv/master/experiments/data/"
    _datasets = {
        "support": "support/support_train_test.h5",
        "metabric": "metabric/metabric_IHC4_clinical_train_test.h5",
        "gbsg": "gbsg/gbsg_cancer_train_test.h5",
    }
    col_duration = "duration"
    col_event = "event"

    def _download(self):
        import h5py

        url = self._dataset_url + self._datasets[self.name]
        path = self.path.parent / f"{self.name}.h5"
        with requests.Session() as s:
            r = s.get(url)
            with open(path, "wb") as f:
                f.write(r.content)

        data = defaultdict(dict)
        with h5py.File(path) as f:
            for ds in f:
                for array in f[ds]:
                    data[ds][array] = f[ds][array][:]

        path.unlink()
        train = _make_df(data["train"])
        test = _make_df(data["test"])
        df = pd.concat([train, test]).reset_index(drop=True)
        df.to_feather(self.path)


def _make_df(data):
    x = data["x"]
    t = data["t"]
    d = data["e"]

    colnames = ["x" + str(i) for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=colnames).assign(duration=t).assign(event=d)
    return df


class _Support(_DatasetDeepSurv):
    """Study to Understand Prognoses Preferences Outcomes and Risks of Treatment (SUPPORT).

    A study of survival for seriously ill hospitalized adults.

    This is the processed data set used in the DeepSurv paper (Katzman et al. 2018), and details
    can be found at https://doi.org/10.1186/s12874-018-0482-1

    See https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data
    for original data.

    Variables:
        x0, ..., x13:
            numerical covariates.
        duration: (duration)
            the right-censored event-times.
        event: (event)
            event indicator {1: event, 0: censoring}.
    """

    name = "support"
    _checksum = "b07a9d216bf04501e832084e5b7955cb84dfef834810037c548dee82ea251f8d"


class _Metabric(_DatasetDeepSurv):
    """The Molecular Taxonomy of Breast Cancer International Consortium (METABRIC).

    Gene and protein expression profiles to determine new breast cancer subgroups in
    order to help physicians provide better treatment recommendations.

    This is the processed data set used in the DeepSurv paper (Katzman et al. 2018), and details
    can be found at https://doi.org/10.1186/s12874-018-0482-1

    See https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data
    for original data.

    Variables:
        x0, ..., x8:
            numerical covariates.
        duration: (duration)
            the right-censored event-times.
        event: (event)
            event indicator {1: event, 0: censoring}.
    """

    name = "metabric"
    _checksum = "310b74b97cc37c9eddd29f253ae3c06015dc63a17a71e4a68ff339dbe265f417"


class _Gbsg(_DatasetDeepSurv):
    """Rotterdam & German Breast Cancer Study Group (GBSG)

    A combination of the Rotterdam tumor bank and the German Breast Cancer Study Group.

    This is the processed data set used in the DeepSurv paper (Katzman et al. 2018), and details
    can be found at https://doi.org/10.1186/s12874-018-0482-1

    See https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data
    for original data.

    Variables:
        x0, ..., x6:
            numerical covariates.
        duration: (duration)
            the right-censored event-times.
        event: (event)
            event indicator {1: event, 0: censoring}.
    """

    name = "gbsg"
    _checksum = "de2359bee62bf36b9e3f901fea4a9fbef2d145e26e9384617d0d3f75892fe5ce"


metabric = _Metabric()
gbsg = _Gbsg()
nwtco = _Nwtco()


import io
import pkgutil

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import torchvision


def _load_framingham_dataset(sequential):
    """Helper function to load and preprocess the Framingham dataset.
    The Framingham Dataset is a subset of 4,434 participants of the well known,
    ongoing Framingham Heart study [1] for studying epidemiology for
    hypertensive and arteriosclerotic cardiovascular disease. It is a popular
    dataset for longitudinal survival analysis with time dependent covariates.
    Parameters
    ----------
    sequential: bool
      If True returns a list of np.arrays for each individual.
      else, returns collapsed results for each time step. To train
      recurrent neural models you would typically use True.
    References
    ----------
    [1] Dawber, Thomas R., Gilcin F. Meadors, and Felix E. Moore Jr.
    "Epidemiological approaches to heart disease: the Framingham Study."
    American Journal of Public Health and the Nations Health 41.3 (1951).
    """
    import auton_survival

    data = os.path.join(
        os.path.dirname(auton_survival.__file__), "datasets/framingham.csv"
    )
    data = pd.read_csv(data)
    data.drop_duplicates(inplace=True, keep="first", subset="RANDID")
    data = data.reset_index(drop=True)

    dat_cat = data[
        [
            "SEX",
            "CURSMOKE",
            "DIABETES",
            "BPMEDS",
            "educ",
            "PREVCHD",
            "PREVAP",
            "PREVMI",
            "PREVSTRK",
            "PREVHYP",
        ]
    ]
    dat_num = data[
        ["TOTCHOL", "AGE", "SYSBP", "DIABP", "CIGPDAY", "BMI", "HEARTRTE", "GLUCOSE"]
    ]

    x1 = pd.get_dummies(dat_cat).values
    x2 = dat_num.values
    x = np.hstack([x1, x2])

    time = (data["TIMEDTH"] - data["TIME"]).values
    event = data["DEATH"].values

    return data.drop(columns=["TIMEDTH", "DEATH"]), time, event


def _load_pbc_dataset(sequential):
    """Helper function to load and preprocess the PBC dataset
    The Primary biliary cirrhosis (PBC) Dataset [1] is well known
    dataset for evaluating survival analysis models with time
    dependent covariates.
    Parameters
    ----------
    sequential: bool
      If True returns a list of np.arrays for each individual.
      else, returns collapsed results for each time step. To train
      recurrent neural models you would typically use True.
    References
    ----------
    [1] Fleming, Thomas R., and David P. Harrington. Counting processes and
    survival analysis. Vol. 169. John Wiley & Sons, 2011.
    """
    import auton_survival

    data = os.path.join(os.path.dirname(auton_survival.__file__), "datasets/pbc2.csv")
    data = pd.read_csv(data)
    data = data.drop_duplicates(keep="first", subset="id")
    data = data.reset_index(drop=True)

    data["histologic"] = data["histologic"].astype(str)
    dat_cat = data[
        ["drug", "sex", "ascites", "hepatomegaly", "spiders", "edema", "histologic"]
    ]
    dat_num = data[
        [
            "serBilir",
            "serChol",
            "albumin",
            "alkaline",
            "SGOT",
            "platelets",
            "prothrombin",
        ]
    ]
    age = data["age"] + data["years"]

    time = (data["years"] - data["year"]).values
    event = data["status2"].values

    return data.drop(columns=["years", "status2"]), time, event
