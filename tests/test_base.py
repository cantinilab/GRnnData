import os

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy.sparse import csr_matrix

from grnndata import GRNAnnData, from_adata_and_longform, utils


def test_base():
    adata = sc.read_h5ad(os.path.join(os.path.dirname(__file__), "test.h5ad"))
    random_matrix = np.random.rand(1000, 1000)
    adata = adata[:, :1000]
    random_mask = np.random.choice([0, 1], size=random_matrix.shape, p=[0.8, 0.2])
    sparse_random_matrix = csr_matrix(random_matrix * random_mask)
    try:
        grn = GRNAnnData(adata.copy(), grn=sparse_random_matrix)
        utils.get_centrality(grn, top_k_to_disp=0)
        subgrn = grn.get("ENSG00000000003")
        assert subgrn.grn.shape == (1, 1), "subgrn shape should be (1, 1)"
        assert subgrn.targets.shape[0] == 1, "subgrn targets shape should be (1, n)"
        assert (
            subgrn.regulators.shape[0] == 1
        ), "subgrn regulators shape should be (1, n)"
        assert isinstance(grn, GRNAnnData), "grn should be an instance of GRNAnnData"
        assert grn.shape == adata.shape, "grn shape should match adata shape"
        print(grn)
        grn.plot_subgraph("TSPAN6")
        sgrn = from_adata_and_longform(
            adata.copy(),
            pd.DataFrame(
                {
                    "regulator": ["ENSG00000000003"],
                    "target": ["ENSG00000000003"],
                }
            ),
        )
        print(sgrn)
        print(sgrn.extract_links())
        metrics = utils.metrics(grn)
        print(metrics)
        del sgrn
        # Test compute_cluster
        utils.compute_connectivities(grn)
        utils.compute_cluster(grn)
        assert (
            "connectivities" in grn.varp.keys()
        ), "connectivities should be added to varp after compute_connectivities"
        assert (
            "cluster_1.5" in grn.var
        ), "clusters should be added to var after compute_cluster"
        # Test compute_enrichment
        grn.var.index = grn.var.symbol
        enrichment_results = utils.enrichment(
            grn,
            gene_sets=[{"TFs": utils.TF}, utils.file_dir + "/celltype.gmt"],
            doplot=False,
        )
        enrichment_results = enrichment_results.res2d
        assert isinstance(
            enrichment_results, pd.DataFrame
        ), "compute_enrichment should return a DataFrame"
        assert not enrichment_results.empty, "enrichment results should not be empty"
    except Exception as e:
        pytest.fail(f"An exception occurred: {str(e)}")
    # assert np.array_equal(grn.X, adata.X), "grn.X should match adata.X"
    # assert np.array_equal(grn.grn.toarray(), sparse_random_matrix.toarray()), "grn.grn should match the input sparse_random_matrix"
