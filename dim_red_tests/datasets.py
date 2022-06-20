from enum import Enum
from os.path import join as pjoin
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn import datasets

from dim_red_tests.utils import get_project_root


def get_s_curve_dataset(
    n_samples: int = 1000, seed: int = 314159
) -> Tuple[pd.DataFrame, pd.Series]:
    data, labels = datasets.make_s_curve(n_samples, random_state=seed)
    # dim1 and dim3 provides the S-shaped 2D curve
    return pd.DataFrame(data=data, columns=["dim1", "dim2", "dim3"]), pd.Series(labels)


def get_circles(
    n_samples: int = 1000, factor: float = 0.5, noise: float = 0.05, seed: int = 314159
) -> Tuple[pd.DataFrame, pd.Series]:
    data, labels = datasets.make_circles(
        n_samples=n_samples, factor=factor, noise=noise, random_state=seed
    )
    return pd.DataFrame(data=data, columns=["dim1", "dim2"]), pd.Series(labels)


def get_uniform(n_samples: int = 1000) -> Tuple[pd.DataFrame, pd.Series]:
    x = np.linspace(0, 1, int(np.sqrt(n_samples)))
    xx, yy = np.meshgrid(x, x)
    data = np.hstack([xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)])
    labels = xx.ravel()
    return pd.DataFrame(data=data, columns=["dim1", "dim2"]), pd.Series(labels)


class FCLabels(Enum):
    NOISE = -1
    DEBRIS = 0
    LYMPH = 1
    MONO = 2
    OTHER = 3


def get_fc_dataset(
    n_samples: int = 1000, seed: int = 314159
) -> Tuple[pd.DataFrame, pd.Series]:
    """ Real flow cytometry dataset example.

    (from colleagues in N. N. Petrov NMRC of Oncology)
    """
    data_path = pjoin(get_project_root(), "data", "flow_cytometry_data.csv")
    df: pd.DataFrame = pd.read_csv(data_path, index_col=0)

    # remove some noise
    flow_mask = (df["FSC-A-"] > 200000) | (df["SSC-A-"] > 240000)
    df = df.drop(df[flow_mask].index).reset_index()

    df = df.sample(n=n_samples, random_state=seed)

    # create labels manually
    conditions = [
        (df["FSC-A-"] < 40000) & (df["SSC-A-"] < 35000),
        (df["FSC-A-"].between(35000, 100000)) & (df["SSC-A-"] < 50000),
        (df["FSC-A-"].between(75000, 150000)) & (df["SSC-A-"].between(50000, 90000)),
        (df["FSC-A-"].between(75000, 200000)) & (df["SSC-A-"].between(90000, 250000)),
    ]
    values = [
        FCLabels.DEBRIS.value,
        FCLabels.LYMPH.value,
        FCLabels.MONO.value,
        FCLabels.OTHER.value,
    ]
    labels = np.select(conditions, values, default=FCLabels.NOISE.value)
    target = pd.Series(labels)

    return df, target
