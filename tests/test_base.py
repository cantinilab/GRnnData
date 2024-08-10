import os
import scanpy as sc
from scipy.sparse import csr_matrix
import numpy as np

from grnndata import GRNAnnData


def test_base():
    adata = sc.read_h5ad(os.path.join(os.path.dirname(__file__), "test.h5ad"))
    random_matrix = np.random.rand(1000, 1000)
    adata = adata[:, :1000]
    random_mask = np.random.choice([0, 1], size=random_matrix.shape, p=[0.8, 0.2])
    sparse_random_matrix = csr_matrix(random_matrix * random_mask)
    grn = GRNAnnData(adata.copy(), grn=sparse_random_matrix)
    assert True, "GRNAnnData test passed"
