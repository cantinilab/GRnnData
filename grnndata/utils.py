import logging
import os.path
from typing import Callable, List

import gseapy as gp
import leidenalg
import louvain
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import powerlaw  # Power laws are probability distributions with the form:p(x)∝x−α
import scanpy as sc
import scipy.sparse
import scipy.stats
from gseapy.plot import barplot, dotplot
from natsort import natsorted
from scanpy import _utils
from scipy.sparse import issparse

from grnndata import GRNAnnData


def fileToList(filename: str, strconv: Callable[[str], str] = lambda x: x) -> List[str]:
    """
    loads an input file with a\\n b\\n.. into a list [a,b,..]
    """
    with open(filename) as f:
        return [strconv(val[:-1]) for val in f.readlines()]


file_dir = os.path.dirname(os.path.realpath(__file__))


TF = fileToList(file_dir + "/TF.txt")
"""
List of transcription factors (TF) loaded from the file 'TF.txt'.
Each line in the file represents a transcription factor.
"""

mTF = fileToList(file_dir + "/mTF.txt")
"""
List of modified transcription factors (mTF) loaded from the file 'mTF.txt'.
Each line in the file represents a modified transcription factor.
"""


def compute_connectivities(grn: GRNAnnData, topK: int = 20) -> GRNAnnData:
    """
    This function computes the connectivities of a given Gene Regulatory Network (GRN).

    It uses the NetworkX library to convert the GRN into a graph and then computes the spanner of the graph.
    The spanner of a graph is a subgraph that approximates the original graph in terms of distances between nodes.
    The stretch parameter determines the maximum distance between nodes in the spanner compared to the original graph.
    The computed connectivities are then stored in the GRN object.

    Args:
        grn : The Gene Regulatory Network for which the connectivities are to be computed.
        stretch : The maximum distance between nodes in the spanner compared to the original graph. Default is 2.

    Returns:
    grn : The Gene Regulatory Network with the computed connectivities.
    """
    grn_mask = grn.grn.apply(lambda row: row >= row.nlargest(topK).min(), axis=1)
    connect = grn.grn.values.copy()
    connect[~grn_mask.values] = 0
    grn.varp["connectivities"] = scipy.sparse.csr_matrix(connect)
    return grn


def get_centrality(
    grn: GRNAnnData, n_largest_to_consider: int = 20, top_k_to_disp: int = 30
) -> list:
    """
    get_centrality uses the networkx library to calculate the centrality of each node in the GRN.

    The centrality is added to the grn object as a new column in the var dataframe.
    also prints the top K most central nodes in the GRN.

    Args:
        grn (GRNAnnData): The gene regulatory network to analyze.
        n_largest_to_consider (int, optional): The number of largest values to consider when constructing the graph. Defaults to 20.
        top_k_to_disp (int, optional): The number of top results to return. Defaults to 30.

    Returns:
        (list): A list of the top K most central genes in the GRN (sorted by centrality
    """
    grn = compute_connectivities(grn, n_largest_to_consider)
    G = nx.from_scipy_sparse_array(grn.varp["connectivities"], create_using=nx.DiGraph)
    centrality = nx.eigenvector_centrality(G)
    grn.var["centrality"] = pd.Series(centrality).values
    top_central_genes = sorted(
        [(gene, centrality) for gene, centrality in grn.var["centrality"].items()],
        key=lambda x: x[1],
        reverse=True,
    )[:top_k_to_disp]

    print("Top central genes:", top_central_genes)
    return top_central_genes


def compute_cluster(
    grn: GRNAnnData,
    resolution: float = 1.5,
    seed: int = 42,
    use: str = "louvain",
    n_iterations: int = 2,
    max_comm_size: int = 0,
):
    """
    compute_cluster uses the igraph library to perform clustering on the GRN.

    Args:
        grn (GRNAnnData): The gene regulatory network to analyze.
        resolution (float, optional): The resolution parameter for the clustering algorithm. Defaults to 1.5.
        seed (int, optional): The random seed for reproducibility. Defaults to 42.
        use (str, optional): The clustering algorithm to use ('louvain' or 'leiden'). Defaults to "louvain".
        n_iterations (int, optional): The number of iterations for the clustering algorithm. Defaults to 2.
        max_comm_size (int, optional): The maximum community size for the clustering algorithm. Defaults to 0.

    Raises:
        ValueError: If connectivities are not found in the GRN object.
        ValueError: If the clustering algorithm specified in 'use' is not 'louvain' or 'leiden'.

    Returns:
        GRNAnnData: The GRNAnnData object with the computed clusters added to the 'var' DataFrame.
    """
    if "connectivities" not in grn.varp:
        raise ValueError("Connectivities not found in the GRN object.")
    g = _utils.get_igraph_from_adjacency(grn.varp["connectivities"], directed=True)
    weights = np.array(g.es["weight"]).astype(np.float64)
    if use == "louvain":
        part = louvain.find_partition(
            g,
            louvain.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            weights=weights,
            seed=seed,
        )
    elif use == "leiden":
        part = leidenalg.find_partition(
            g,
            leidenalg.ModularityVertexPartition,
            # resolution_parameter=resolution,
            n_iterations=n_iterations,
            weights=weights,
            max_comm_size=max_comm_size,
            seed=seed,
        )
    else:
        raise ValueError("use must be one of 'louvain' or 'leiden'")
    groups = np.array(part.membership)
    grn.var["cluster_" + f"{resolution:.1f}"] = pd.Categorical(
        values=groups.astype("U"),
        categories=natsorted(map(str, np.unique(groups))),
    )
    return grn


def enrichment(
    grn: GRNAnnData,
    of: str = "Targets",
    doplot: bool = True,
    top_k: int = 30,
    gene_sets: list = [
        "KEGG_2016",
        "ENCODE_TF_ChIP-seq_2014",
        "GO_Molecular_Function_2015",
        {"TFs": TF},
        file_dir + "/celltype.gmt",
        "OMIM_Disease",
        "WikiPathways_2016",
        "GO_Cellular_Component_2015",
        "GTEx_Tissue_Sample_Gene_Expression_Profiles_up",
        "TargetScan_microRNA",
        "Chromosome_Location",
        "PPI_Hub_Proteins",
    ],
    min_size: int = 3,
    max_size: int = 4000,
    permutation_num: int = 1000,  # reduce number to speed up testing
    **kwargs,
):
    """
    This function performs enrichment analysis on a given gene regulatory network (grn).

    Args:
        grn (GRNAnnData): The gene regulatory network to analyze.
        of (str, optional): The specific component of the grn to focus on.
        top_k (int, optional): The number of top results to return. Defaults to 10.
        doplot (bool, optional): Whether to generate a plot of the results. Defaults to False.
        gene_sets (list, optional): The gene sets to use for the enrichment analysis. Defaults to the gene sets provided in the function.
        min_size (int, optional): The minimum size of the gene sets to consider. Defaults to 3.
        max_size (int, optional): The maximum size of the gene sets to consider. Defaults to 4000.
        permutation_num (int, optional): The number of permutations to use for the enrichment analysis. Defaults to 1000.
        **kwargs: Additional keyword arguments to be passed to the gseapy.prerank function.

    Returns:
        pandas.DataFrame: A DataFrame containing the results of the enrichment analysis.
    """
    if of == "Targets":
        rnk = grn.grn.sum(1).sort_values(ascending=False)
    elif of == "Regulators":
        rnk = grn.grn.sum(0).sort_values(ascending=False)
    elif of == "Central":
        try:
            get_centrality(grn, top_k_to_disp=0)
        except nx.PowerIterationFailedConvergence:
            print("PowerIterationFailedConvergence")
            return None
        rnk = grn.var["centrality"].sort_values(ascending=False)
    else:
        raise ValueError("of must be one of 'Targets', 'Regulators', or 'Central'")
    rnk.name = None
    rnk = rnk[rnk != 0]
    if rnk.nunique() == 1:
        print("The DataFrame contains only the same values.")
        return None
    # run enrichment analysis
    previous_level = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        pre_res = gp.prerank(
            rnk=rnk,  # or rnk = rnk,
            gene_sets=gene_sets,
            min_size=min_size,
            max_size=max_size,
            background=grn.var.index.tolist(),  # or "hsapiens_gene_ensembl", or int, or text file  or a list of genes
            permutation_num=permutation_num,
            **kwargs,
        )
        logging.disable(previous_level)
    except LookupError:
        print("raised a lookup error")
        return None
    val = (
        pre_res.res2d[(pre_res.res2d["FDR q-val"] < 0.1) & (pre_res.res2d["NES"] > 1)]
        .sort_values(by=["NES"], ascending=False)
        .drop(columns=["Name"])
    )

    # plot results
    if doplot:
        print(val.Term.tolist()[:top_k])
        _ = dotplot(
            pre_res.res2d[
                (pre_res.res2d["FDR q-val"] < 0.1) & (pre_res.res2d["NES"] > 1)
            ].sort_values(by=["NES"], ascending=False),
            column="FDR q-val",
            title="enrichment of " + of + " in the grn",
            size=6,  # adjust dot size
            figsize=(4, 5),
            cutoff=0.25,
            show_ring=False,
        )

    return pre_res


def enrichment_of(grn, target, of="Targets", doplot=False):
    enr = gp.enrich(
        gene_list=grn.var[
            (grn.get(target).varm[of] != 0)[0]
        ].index.tolist(),  # or gene_list=glist
        gene_sets=[
            "KEGG_2016",
            "ENCODE_TF_ChIP-seq_2014",
            "GO_Molecular_Function_2015",
            {"TFs": TF},
            file_dir + "/celltype.gmt",
            "OMIM_Disease",
            "WikiPathways_2016",
            "GO_Cellular_Component_2015",
            "GTEx_Tissue_Sample_Gene_Expression_Profiles_up",
            "TargetScan_microRNA",
            "Chromosome_Location",
            "PPI_Hub_Proteins",
        ],
        background=grn.var.index.tolist(),  # or "hsapiens_gene_ensembl", or int, or text file, or a list of genes
        outdir=None,
        verbose=True,
    )
    if doplot:
        _ = dotplot(
            enr.res2d[
                (enr.res2d["FDR q-val"] < 0.1) & (enr.res2d["NES"] > 1)
            ].sort_values(by=["NES"], ascending=False),
            column="FDR q-val",
            title="enrichment of " + of + " in the grn",
            size=6,  # adjust dot size
            figsize=(4, 5),
            cutoff=0.25,
            show_ring=False,
        )
    return enr.results[enr.results["Adjusted P-value"] < 0.2]


def metrics(grn):
    """
    metrics This function computes the following metrics for a given Gene Regulatory Network (GRN):

    ### small worldness
    A small-world network is a type of mathematical graph in which most nodes are not neighbors of one another,
    but most nodes can be reached from every other by a small number of hops or steps.
    the sigma metric is a measure of small-worldness.
    if sigma > 1, the network is a small-world network.

    ### scale freeness
    A scale-free network is a network whose degree distribution follows a power law,
    at least asymptotically. The s metric is a measure of scale-freeness,
    defined as ( S(G)={frac {s(G)}{s_{max }}} ), where ( s_{max } )
    is the maximum value of s(H) for H in the set of all graphs with degree distribution.
    A graph with small S(G) is "scale-rich," and a graph with S(G) close to 1 is "scale-free"

    Therefore, a graph is more scale-free when its S(G) value is closer to 1.
    The range of the s metric is between 0 and 1, where 0 indicates "scale-rich" and 1 indicates "scale-free"

    Args:
        grn : The Gene Regulatory Network for which the connectivities are to be computed.

    Returns:
        dict : {is_connected: bool, scale_freeness: float}
    """
    G = nx.from_numpy_array(grn.varp["GRN"])
    # sw_sig = nx.sigma(G) # commented because too long
    conn = nx.is_connected(G)
    # clust_coef = nx.average_clustering(G)
    degree_sequence = sorted(
        [d for n, d in G.degree()], reverse=True
    )  # used for degree distribution and powerlaw test
    fit = powerlaw.Fit(degree_sequence)
    R, p = fit.distribution_compare("power_law", "exponential", normalized_ratio=True)
    _ = fit.plot_pdf(color="b", linewidth=2)
    plt.figure(figsize=(10, 6))
    fit.distribution_compare("power_law", "lognormal")
    fig4 = fit.plot_ccdf(linewidth=3, color="black")
    fit.power_law.plot_ccdf(ax=fig4, color="r", linestyle="--")  # powerlaw
    fit.lognormal.plot_ccdf(ax=fig4, color="g", linestyle="--")  # lognormal
    fit.stretched_exponential.plot_ccdf(
        ax=fig4, color="b", linestyle="--"
    )  # stretched_exponential
    return {
        # "small_worldness": sw_sig,
        "is_connected": conn,
        #  "clustering": clust_coef,
        "scale_freeness": [R, p],
    }


def plot_cluster(grn, color=["louvain"], min_dist=0.5, spread=0.7, stretch=2, **kwargs):
    """
    This function plots the clusters of a given Gene Regulatory Network (GRN).

    It first computes the connectivities of the GRN and then performs Louvain clustering on the transpose of the GRN.
    The clusters are then visualized using UMAP.

    Args:
        grn : The Gene Regulatory Network to be clustered and visualized.
        color : The color of the clusters. Default is "louvain".
        min_dist : The minimum distance between points in the UMAP. Default is 0.5.
        spread : The spread of the points in the UMAP. Default is 0.7.
        stretch : The maximum distance between nodes in the spanner compared to the original graph. Default is 2.
        **kwargs : Additional keyword arguments to be passed to the UMAP function.

    """
    grn = compute_connectivities(grn, stretch=stretch)
    subgrn = grn.T
    sc.tl.louvain(subgrn, adjacency=subgrn.obsp["GRN"])

    sc.pl.umap(subgrn, color=color, **kwargs)
