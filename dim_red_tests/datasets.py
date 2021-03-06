from enum import Enum
from os.path import join as pjoin
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn import datasets

from dim_red_tests.utils import PROJECT_ROOT

DatasetT = Tuple[pd.DataFrame, pd.Series]

COLUMNS_2D = ["dim1", "dim2"]
COLUMNS_3D = ["dim1", "dim2", "dim3"]


def get_datasets() -> Dict[str, DatasetT]:
    return {
        "S-curve, 3D": get_s_curve_dataset(),
        "Circles, 2D": get_circles(),
        "Uniform, 2D": get_uniform(),
        "Flow Cytometry, 10D": get_fc_dataset(),
    }


def get_s_curve_dataset(n_samples: int = 1200, seed: int = 314159) -> DatasetT:
    data, labels = datasets.make_s_curve(n_samples, random_state=seed)

    # dim1 and dim3 provides the S-shaped 2D curve
    df, target = pd.DataFrame(data=data, columns=COLUMNS_3D), pd.Series(labels)

    # make a hole
    mask = df['dim1'] ** 2 + (df['dim2'] - 1) ** 2 >= 0.2
    df = df[mask]
    target = target[mask]

    return df, target


def get_circles(
    n_samples: int = 1000, factor: float = 0.5, noise: float = 0.05, seed: int = 314159
) -> DatasetT:
    data, labels = datasets.make_circles(
        n_samples=n_samples, factor=factor, noise=noise, random_state=seed
    )
    return pd.DataFrame(data=data, columns=COLUMNS_2D), pd.Series(labels)


def get_uniform(n_samples: int = 1000) -> DatasetT:
    x = np.linspace(0, 1, int(np.sqrt(n_samples)))
    xx, yy = np.meshgrid(x, x)
    data = np.hstack([xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)])
    labels = xx.ravel()
    return pd.DataFrame(data=data, columns=COLUMNS_2D), pd.Series(labels)


class FCLabels(Enum):
    NOISE = -1
    DEBRIS = 0
    LYMPH = 1
    MONO = 2
    OTHER = 3


COLUMNS_FC = [
    "FSC-A-",
    "SSC-A-",
    "FITC-A-CD25",
    "PE-A-CD127",
    "PerCP-Cy5-5-A-CD4",
    "PE-Cy7-A-",
    "APC-A-",
    "APC-Cy7-A-",
    "Pacific Blue-A-",
    "AmCyan-A-",
]

COLUMNS_FC_MAIN = ["FSC-A-", "SSC-A-"]


def get_fc_dataset(n_samples: int = 1000, seed: int = 314159) -> DatasetT:
    """Real flow cytometry dataset example.

    (from colleagues in N. N. Petrov NMRC of Oncology)
    """
    data_path = pjoin(PROJECT_ROOT, "data", "flow_cytometry_data.csv")
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

    df = df.drop(columns="Time-")

    return df, target


def add_noisy_columns(
    df: pd.DataFrame, n_noisy_cols: int = 0, normal_noise_over_uniform: bool = True
):
    if not n_noisy_cols:
        return df

    df = df.join(
        pd.DataFrame(
            {
                f"_noise_normal_{i}": np.random.normal(0, 1, size=df.shape[0])
                if normal_noise_over_uniform
                else np.random.random(size=df.shape[0])
                for i in range(n_noisy_cols)
            },
            index=df.index,
        )
    )
    return df
