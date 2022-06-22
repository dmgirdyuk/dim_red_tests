from os.path import join as pjoin
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from matplotlib.axes import Axes
from sklearn.preprocessing import StandardScaler

from dim_red_tests.datasets import FCLabels
from dim_red_tests.utils import RESULTS_FOLDER


def plot_2d(
    data: pd.DataFrame,
    target: pd.Series,
    fig_size: Tuple[int, int] = (8, 8),
    title: str = "",
    show: bool = False,
    save: bool = True,
    filename: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=fig_size, facecolor="white", constrained_layout=True)

    fig.suptitle(title, size=16)
    add_2d_scatter(ax, data, target)

    if show:
        plt.show()

    if save:
        assert filename is not None, "Specify filename to save plot."
        fig.savefig(pjoin(RESULTS_FOLDER, f"{filename}.png"), dpi=150)


def add_2d_scatter(
    ax: Axes,
    df: pd.DataFrame,
    target: pd.Series,
    marker_size: int = 50,
    title: Optional[str] = None,
) -> None:
    x, y = df.iloc[:, 0], df.iloc[:, 1]

    ax.scatter(x, y, c=target, s=marker_size, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())


def plot_3d(
    df: pd.DataFrame,
    target: pd.Series,
    fig_size: Tuple[int, int] = (8, 8),
    title: str = "",
    show: bool = False,
    save: bool = True,
    filename: Optional[str] = None,
) -> None:
    x, y, z = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]

    fig, ax = plt.subplots(
        figsize=fig_size,
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)

    col = ax.scatter(x, y, z, c=target, s=50, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.colorbar(col, ax=ax, orientation="vertical", shrink=0.5, aspect=30, pad=0.01)

    if show:
        plt.show()

    if save:
        assert filename is not None, "Specify filename to save plot."
        fig.savefig(pjoin(RESULTS_FOLDER, f"{filename}.png"), dpi=150)


def plot_pairplot_fc(df: pd.DataFrame, target: pd.Series, filename: str) -> None:
    df_new = df.copy()
    labels = target.apply(lambda x: FCLabels(x).name.lower().capitalize())
    df_new = df_new[df_new.columns[:5]]
    df_new[df_new.columns] = StandardScaler().fit_transform(df_new)
    df_new["labels"] = labels.values
    sns.pairplot(df_new, hue="labels", diag_kind="hist").figure.savefig(
        pjoin(RESULTS_FOLDER, f"{filename}.png"), dpi=150
    )
