from __future__ import annotations

from enum import Enum
from typing import Union

from sklearn.decomposition import PCA
from sklearn.manifold import (
    MDS,
    TSNE,
    Isomap,
    LocallyLinearEmbedding,
    SpectralEmbedding,
)
from umap import UMAP

DimRedMethodT = Union[
    MDS, TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding, UMAP
]


def get_pca(n_components: int = 2) -> PCA:
    return PCA(n_components=n_components)


def get_mds(
    n_components: int = 2, n_init: int = 1, max_iter: int = 120, n_jobs: int = 4
) -> MDS:
    return MDS(
        n_components=n_components, n_init=n_init, max_iter=max_iter, n_jobs=n_jobs
    )


def get_isomap(n_components: int = 2, n_neighbors: int = 20, n_jobs: int = 4) -> Isomap:
    return Isomap(n_components=n_components, n_neighbors=n_neighbors, n_jobs=n_jobs)


class LLEMethods(Enum):
    STANDARD = "standard"
    HESSIAN = "hessian"
    MODIFIED = "modified"
    LTSA = "ltsa"


def get_lle(
    n_components: int = 2,
    n_neighbors: int = 20,
    method: LLEMethods = LLEMethods.STANDARD,
    seed: int = 314159,
    n_jobs: int = 4,
) -> LocallyLinearEmbedding:
    return LocallyLinearEmbedding(
        n_components=n_components,
        n_neighbors=n_neighbors,
        method=method.value,
        random_state=seed,
        n_jobs=n_jobs,
    )


def get_spectral_embedding(
    n_components: int = 2,
    n_neighbors: int = 20,
    seed: int = 314159,
    n_jobs: int = 4,
) -> SpectralEmbedding:
    return SpectralEmbedding(
        n_components=n_components,
        n_neighbors=n_neighbors,
        random_state=seed,
        n_jobs=n_jobs,
    )


def get_tsne(
    n_components: int = 2,
    perplexity: int = 50,
    n_iter: int = 500,
    init: str = "pca",
    seed: int = 314159,
    n_jobs: int = 4,
) -> TSNE:
    return TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate="auto",
        n_iter=n_iter,
        init=init,
        random_state=seed,
        n_jobs=n_jobs,
    )


def get_umap(
    n_components: int = 2,
    n_neighbors: int = 20,
    min_dist: float = 0.5,
    seed: int = 314159,
    n_jobs: int = 4,
) -> UMAP:
    return UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
        n_jobs=n_jobs,
    )
