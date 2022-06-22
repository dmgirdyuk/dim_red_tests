from os.path import join as pjoin
from typing import Dict

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from dim_red_tests.algorithms import (
    LLEMethods,
    get_isomap,
    get_lle,
    get_mds,
    get_pca,
    get_spectral_embedding,
    get_tsne,
    get_umap,
)
from dim_red_tests.datasets import (
    COLUMNS_2D,
    COLUMNS_3D,
    COLUMNS_FC_MAIN,
    DatasetT,
    add_noisy_columns,
    get_datasets,
    get_s_curve_dataset,
)
from dim_red_tests.utils import RESULTS_FOLDER, seed_everything, timeit
from dim_red_tests.visualization import (
    add_2d_scatter,
    plot_2d,
    plot_3d,
    plot_pairplot_fc,
)

DATASETS = get_datasets()


@timeit
def plot_all_datasets():
    df, target = DATASETS["Flow Cytometry, 10D"]
    plot_2d(
        df[COLUMNS_FC_MAIN], target, title="Flow Cytometry, 10D", filename="dataset_fc"
    )
    plot_pairplot_fc(df, target, filename="dataset_fc_pairplot")

    df, target = DATASETS["S-curve, 3D"]
    plot_3d(df, target, title="S-curve", filename="dataset_s_curve")

    df, target = DATASETS["Circles, 2D"]
    plot_2d(df, target, title="Circles", filename="dataset_circles")

    df, target = DATASETS["Uniform, 2D"]
    plot_2d(df, target, title="Uniform", filename="dataset_uniform")


@timeit
def run_experiment_pca(seed: int = 314159):
    """PCA."""
    seed_everything(seed)
    df, target = get_s_curve_dataset()

    def _run_experiment(
        title: str,
        filename: str,
        n_noisy_cols: int = 0,
        with_mean: bool = True,
        with_std: bool = False,
    ) -> None:
        df_test = df.copy()
        if n_noisy_cols:
            df_test = add_noisy_columns(df_test, n_noisy_cols)

        scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        df_test[COLUMNS_3D] = scaler.fit_transform(df_test[COLUMNS_3D])

        pca = get_pca(n_components=2)
        df_embedding = pd.DataFrame(
            columns=COLUMNS_2D, data=pca.fit_transform(df_test, target)
        )

        plot_2d(df_embedding, target, title=title, filename=filename)

    _run_experiment(with_std=False, title="PCA, minus mean", filename="exp_1_pca_1")
    _run_experiment(with_std=True, title="PCA, normalized", filename="exp_1_pca_2")
    _run_experiment(
        with_std=False,
        n_noisy_cols=7,
        title="PCA, minus mean + 7 noisy columns",
        filename="exp_1_pca_3",
    )


@timeit
def run_experiment_comparison(
    add_noise: bool = False, normal_noise_over_uniform: bool = False, seed: int = 314159
):
    seed_everything(seed)

    methods = {
        "PCA": get_pca(),
        "MDS": get_mds(),
        "Isomap": get_isomap(),
        "LLE": get_lle(),
        "Modified LLE": get_lle(method=LLEMethods.MODIFIED),
        "Spectral": get_spectral_embedding(),
        "t-SNE": get_tsne(),
        "UMAP": get_umap(),
    }
    datasets = prepare_datasets_for_comparison(add_noise, normal_noise_over_uniform)

    fig, axes = plt.subplots(len(DATASETS), len(methods) + 1, figsize=(18, 8))
    for i, (dataset_name, (df, target)) in enumerate(datasets.items()):
        add_2d_scatter(
            axes[i, 0],
            df.iloc[:, :2],
            target,
            marker_size=10,
            title=dataset_name,
        )
        for j, (method_name, method) in enumerate(methods.items()):
            df_embedding = pd.DataFrame(
                columns=COLUMNS_2D, data=method.fit_transform(df, target)
            )
            add_2d_scatter(
                axes[i, j + 1], df_embedding, target, marker_size=5, title=method_name
            )

    noise_type = "normal" if normal_noise_over_uniform else "uniform"
    dataset_type = "general" if not add_noise else noise_type
    filename = f"exp_2_{dataset_type}.png"
    fig.savefig(pjoin(RESULTS_FOLDER, filename), dpi=150)


@timeit
def prepare_datasets_for_comparison(
    add_noise: bool = False, normal_noise_over_uniform: bool = False
) -> Dict[str, DatasetT]:
    datasets = {}
    scaler = StandardScaler()
    for dataset_name, (df, target) in DATASETS.items():
        df_new = df.copy()
        if add_noise:
            df_new = add_noisy_columns(
                df_new,
                n_noisy_cols=15 - df_new.shape[1],
                normal_noise_over_uniform=normal_noise_over_uniform,
            )
        df_new[df_new.columns] = scaler.fit_transform(df_new)

        if dataset_name == "S-curve, 3D":
            df_new[["dim2", "dim3"]] = df_new[["dim3", "dim2"]]

        datasets[dataset_name] = (df_new, target)

    return datasets


@timeit
def main():
    plot_all_datasets()
    run_experiment_pca()
    run_experiment_comparison(add_noise=False)
    run_experiment_comparison(add_noise=True, normal_noise_over_uniform=True)
    run_experiment_comparison(add_noise=True, normal_noise_over_uniform=False)


if __name__ == "__main__":
    main()
