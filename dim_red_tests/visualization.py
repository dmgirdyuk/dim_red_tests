from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker
from matplotlib.axes import Axes


def plot_2d(data: pd.DataFrame, target: pd.Series, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white", constrained_layout=True)

    fig.suptitle(title, size=16)
    add_2d_scatter(ax, data, target)

    plt.show()


def add_2d_scatter(
    ax: Axes, data: pd.DataFrame, target: pd.Series, title: Optional[str] = None
) -> None:
    x, y = data.iloc[:, 0], data.iloc[:, 1]

    ax.scatter(x, y, c=target, s=50, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())


def plot_3d(
    data: pd.DataFrame,
    target: pd.Series,
    fig_size: Tuple[int, int] = (8, 8),
    title: str = "",
) -> None:
    x, y, z = data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2]

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

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)

    plt.show()
