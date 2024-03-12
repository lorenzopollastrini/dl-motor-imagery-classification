import numpy as np
from scipy.sparse import coo, coo_matrix, csr_matrix, find, hstack, vstack


def coarsen(adjacency_matrix, levels, self_connections=False):
    """Coarsen a graph, represented by its adjacency matrix, at multiple levels."""
    graphs, parents = metis(adjacency_matrix, levels)
    permutations = compute_permutations(parents)

    for i, adjacency_matrix in enumerate(graphs):
        n_nodes, _ = adjacency_matrix.shape

        if not self_connections:
            adjacency_matrix = adjacency_matrix.tocoo()
            adjacency_matrix.setdiag(0)

        if i < levels:
            adjacency_matrix = permute_adjacency(adjacency_matrix, permutations[i])

        adjacency_matrix = adjacency_matrix.tocsr()
        adjacency_matrix.eliminate_zeros()
        graphs[i] = adjacency_matrix

    return graphs, permutations[0] if levels > 0 else None


def metis(adjacency_matrix, levels, rid=None):
    """Coarsen a graph, represented by its adjacency matrix, at multiple levels, using the METIS algorithm.

    INPUT
    adjacency_matrix: the adjacency matrix of the graph
    levels: the number of coarsened graphs

    OUTPUT
    graphs[0]: original graph of size N_1
    graphs[2]: coarser graph of size N_2 < N_1
    graphs[levels]: coarsest graph of size N_levels < ... < N_2 < N_1
    parents[i]: vector of size N_i with entries ranging from 1 to N_{i+1} which indicate the parents in the coarser
        graph graphs[i+1]

    NOTE
    If "graphs" is a list of length k, then "parents" will be a list of length k-1
    """

    n_nodes, _ = adjacency_matrix.shape
    if rid is None:
        rid = np.random.permutation(range(n_nodes))
    parents = []
    degrees = adjacency_matrix.sum(axis=0) - adjacency_matrix.diagonal()
    graphs = [adjacency_matrix]

    for _ in range(levels):
        # Weights for the pairing
        # weights = ones(n_nodes,1)  # metis weights
        weights = degrees  # graclus weights
        weights = np.array(weights).squeeze()

        # Pair the nodes and construct the root vector
        idx_row, idx_col, val = find(adjacency_matrix)
        permutations = np.argsort(idx_row)
        rr = idx_row[permutations]
        cc = idx_col[permutations]
        vv = val[permutations]
        cluster_id = metis_one_level(rr, cc, vv, rid, weights)  # rr is ordered
        parents.append(cluster_id)

        # Compute the edges weights for the new graph
        nrr = cluster_id[rr]
        ncc = cluster_id[cc]
        nvv = vv
        n_new = np.max(cluster_id) + 1
        # CSR is more appropriate: row, val pairs appear multiple times
        adjacency_matrix = csr_matrix((nvv, (nrr, ncc)), shape=(n_new, n_new))
        adjacency_matrix.eliminate_zeros()
        # Add new graph to the list of all coarsened graphs
        graphs.append(adjacency_matrix)
        n_nodes, _ = adjacency_matrix.shape

        # Compute the degrees
        degrees = adjacency_matrix.sum(axis=0)
        # degrees = W.sum(axis=0) - W.diagonal()

        # Choose the order in which nodes will be visited at the next pass
        ss = np.array(adjacency_matrix.sum(axis=0)).squeeze()
        rid = np.argsort(ss)
    return graphs, parents


def metis_one_level(rr, cc, vv, rid, weights):
    """Coarsen a graph given by rr, cc, and vv, using the METIS algorithm. rr is assumed to be ordered."""

    nnz = rr.shape[0]
    n = rr[nnz-1] + 1

    marked = np.zeros(n, np.bool_)
    row_start = np.zeros(n, np.int32)
    row_length = np.zeros(n, np.int32)
    cluster_id = np.zeros(n, np.int32)

    old_val = rr[0]
    count = 0
    cluster_count = 0

    for ii in range(nnz):
        row_length[count] = row_length[count] + 1
        if rr[ii] > old_val:
            old_val = rr[ii]
            row_start[count+1] = ii
            count = count + 1

    for ii in range(n):
        tid = rid[ii]
        if not marked[tid]:
            w_max = 0.0
            rs = row_start[tid]
            marked[tid] = True
            best_neighbor = -1
            for jj in range(row_length[tid]):
                nid = cc[rs+jj]

                if marked[nid]:
                    t_val = 0.0
                else:
                    t_val = vv[rs+jj] * (1.0/weights[tid] + 1.0/weights[nid])

                if t_val > w_max:
                    w_max = t_val
                    best_neighbor = nid

            cluster_id[tid] = cluster_count

            if best_neighbor > -1:
                cluster_id[best_neighbor] = cluster_count
                marked[best_neighbor] = True

            cluster_count += 1
    return cluster_id


def compute_permutations(parents):
    """Return a list of indices to reorder the adjacency and data matrices so that merging neighbors in pairs forms a
    binary tree from layer to layer.
    """
    # Order of last layer is random (chosen by the clustering algorithm)
    indices = []
    if len(parents) > 0:
        n_nodes_last = max(parents[-1]) + 1
        indices.append(list(range(n_nodes_last)))

        for parent in parents[::-1]:
            # Fake nodes go after real ones
            pool_singletons = len(parent)

            indices_layer = []
            for i in indices[-1]:
                indices_node = list(np.where(parent == i)[0])
                assert 0 <= len(indices_node) <= 2

                # Add a node to go with a singleton
                if len(indices_node) == 1:
                    indices_node.append(pool_singletons)
                    pool_singletons += 1
                # Add two nodes as children of a singleton in the parent
                elif len(indices_node) == 0:
                    indices_node.append(pool_singletons+0)
                    indices_node.append(pool_singletons+1)
                    pool_singletons += 2

                indices_layer.extend(indices_node)
            indices.append(indices_layer)

        # Sanity checks
        for i, indices_layer in enumerate(indices):
            n_nodes = n_nodes_last*2**i
            # Make sure that there's a reduction by a factor of 2 at each layer (binary tree)
            assert len(indices[0] == n_nodes)
            # Make sure that the new ordering does not omit any index
            assert sorted(indices_layer) == list(range(n_nodes))

    return indices[::-1]


def permute_data(data_matrix, indices):
    """Permute data matrix, i.e. exchange node IDs, so that merging neighbors in pairs forms a binary tree from layer to
    layer.
    """
    if indices is None:
        return data_matrix

    n_samples, n_nodes = data_matrix.shape
    n_nodes_new = len(indices)
    assert n_nodes_new >= n_nodes
    data_matrix_new = np.empty((n_samples, n_nodes_new))
    for i, j in enumerate(indices):
        # Existing node, i.e. real data
        if j < n_nodes:
            data_matrix_new[:, i] = data_matrix[:, j]
        # Fake node because of singletons. They will stay 0 so that max pooling chooses the singleton
        else:
            data_matrix_new[:, i] = np.zeros(n_samples)
    return data_matrix_new


def permute_adjacency(adjacency_matrix, indices):
    """Permute adjacency matrix, i.e. exchange node IDs, so that merging neighbors in pairs forms a binary tree from
    layer to layer.
    """
    if indices is None:
        return adjacency_matrix

    n_nodes, _ = adjacency_matrix.shape
    n_nodes_new = len(indices)
    assert n_nodes_new >= n_nodes
    adjacency_matrix = adjacency_matrix.tocoo()

    # Add n_nodes_new - n_nodes isolated nodes
    if n_nodes_new > n_nodes:
        rows = coo_matrix((n_nodes_new-n_nodes, n_nodes), dtype=np.float32)
        cols = coo_matrix((n_nodes_new, n_nodes_new-n_nodes), dtype=np.float32)
        adjacency_matrix = vstack([adjacency_matrix, rows])
        adjacency_matrix = hstack([adjacency_matrix, cols])

    # Permute the rows and the columns
    permutations = np.argsort(indices)
    adjacency_matrix.row = np.array(permutations)[adjacency_matrix.row]
    adjacency_matrix.col = np.array(permutations)[adjacency_matrix.col]

    assert type(adjacency_matrix) is coo.coo_matrix
    return adjacency_matrix
