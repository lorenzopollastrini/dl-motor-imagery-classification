import numpy as np
from scipy.sparse import csr, diags, identity


def laplacian(adjacency_matrix, normalized=True):
    degrees = adjacency_matrix.sum(axis=0)

    if not normalized:
        degree_matrix = diags(degrees.adjacency_matrix.squeeze(), 0)
        laplacian = degree_matrix - adjacency_matrix
    else:
        degrees += np.spacing(np.array(0, adjacency_matrix.dtype))
        degrees = 1 / np.sqrt(degrees)
        degree_matrix = diags(degrees.A.squeeze(), 0)
        identity_matrix = identity(degrees.size, dtype=adjacency_matrix.dtype)
        laplacian = identity_matrix - degree_matrix * adjacency_matrix * degree_matrix

    assert type(laplacian) is csr.csr_matrix

    return laplacian


def normalize_laplacian(laplacian, max_eigenvalue):
    """Rescale the Laplacian eigenvalues in [-1, 1]."""
    n_nodes, _ = laplacian.shape
    identity_matrix = identity(n_nodes, format='csr', dtype=laplacian.dtype)

    laplacian /= max_eigenvalue / 2
    laplacian -= identity_matrix

    return laplacian
