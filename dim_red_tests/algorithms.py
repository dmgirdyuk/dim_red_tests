from __future__ import annotations

from enum import Enum

from sklearn.decomposition import PCA
from sklearn.manifold import (
    MDS,
    TSNE,
    Isomap,
    LocallyLinearEmbedding,
    SpectralEmbedding,
)
from umap import UMAP


def get_pca(n_components: int = 2) -> PCA:
    return PCA(n_components=n_components)


def get_mds(
    n_components: int = 2, n_init: int = 1, max_iter: int = 120, n_jobs: int = 4
) -> MDS:
    return MDS(
        n_components=n_components, n_init=n_init, max_iter=max_iter, n_jobs=n_jobs
    )


def get_isomap(n_components: int = 2, n_neighbors: int = 10, n_jobs: int = 4) -> Isomap:
    return Isomap(n_components=n_components, n_neighbors=n_neighbors, n_jobs=n_jobs)


class LLEMethods(Enum):
    STANDARD = "standard"
    HESSIAN = "hessian"
    MODIFIED = "modified"
    LTSA = "ltsa"


def get_lle(
    n_components: int = 2,
    n_neighbors: int = 10,
    method: str = LLEMethods.STANDARD.name,
    seed: int = 314159,
    n_jobs: int = 4,
) -> LocallyLinearEmbedding:
    return LocallyLinearEmbedding(
        n_components=n_components,
        n_neighbors=n_neighbors,
        method=method,
        random_state=seed,
        n_jobs=n_jobs,
    )


def get_spectral_embedding(
    n_components: int = 2, seed: int = 314159, n_jobs: int = 4
) -> SpectralEmbedding:
    return SpectralEmbedding(
        n_components=n_components, random_state=seed, n_jobs=n_jobs
    )


def get_tsne(
    n_components: int = 2,
    perplexity: int = 30,
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
    n_neighbors: int = 10,
    min_dist: float = 0.1,
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


EMBEDDING_GETTERS = {
    "PCA": get_pca,
    "MDS embedding": get_mds,
    "Isomap embedding": get_isomap,
    "LLE embedding": get_lle,
    "Laplacian Eigenmaps": get_spectral_embedding,
    "t-SNE embeedding": get_tsne,
    "UMAP": get_umap,
}
