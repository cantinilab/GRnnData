import os

import numpy as np
import pytest
import scanpy as sc
from scipy.sparse import csr_matrix

from grnndata import GRNAnnData, utils


def test_base():
    adata = sc.read_h5ad(os.path.join(os.path.dirname(__file__), "test.h5ad"))
    random_matrix = np.random.rand(1000, 1000)
    adata = adata[:, :1000]
    random_mask = np.random.choice([0, 1], size=random_matrix.shape, p=[0.8, 0.2])
    sparse_random_matrix = csr_matrix(random_matrix * random_mask)
    try:
        grn = GRNAnnData(adata.copy(), grn=sparse_random_matrix)
        utils.get_centrality(grn, top_k_to_disp=0)
        assert isinstance(grn, GRNAnnData), "grn should be an instance of GRNAnnData"
        assert grn.shape == adata.shape, "grn shape should match adata shape"
    except Exception as e:
        pytest.fail(f"An exception occurred: {str(e)}")
    # assert np.array_equal(grn.X, adata.X), "grn.X should match adata.X"
    # assert np.array_equal(grn.grn.toarray(), sparse_random_matrix.toarray()), "grn.grn should match the input sparse_random_matrix"
