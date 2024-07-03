import networkx as nx
import gseapy as gp
from gseapy.plot import barplot, dotplot
import numpy as np
import os.path
import scanpy as sc
from scipy.sparse import issparse
import scipy
import powerlaw  # Power laws are probability distributions with the form:p(x)∝x−α
import logging


def fileToList(filename, strconv=lambda x: x):
    """
    loads an input file with a\\n b\\n.. into a list [a,b,..]
    """
    with open(filename) as f:
        return [strconv(val[:-1]) for val in f.readlines()]


file_dir = os.path.dirname(os.path.realpath(__file__))
TF = fileToList(file_dir + "/TF.txt")
mTF = fileToList(file_dir + "/mTF.txt")


def get_centrality(grn, top_k=30):
    """
    get_centrality uses the networkx library to calculate the centrality of each node in the GRN.
    The centrality is added to the grn object as a new column in the var dataframe.
    also prints the top K most central nodes in the GRN.

    Parameters:
        grn (GRNAnnData): The gene regulatory network to analyze.
        top_k (int, optional): The number of top results to return. Defaults to 30.

    Returns:
        (list): A list of the top K most central genes in the GRN (sorted by centrality
    """
    G = nx.from_numpy_array(grn.varp["GRN"])
    centrality = nx.eigenvector_centrality(G)

    grn.var["centrality"] = [
        centrality.get(gene, 0) for gene in range(len(grn.var_names))
    ]

    top_central_genes = sorted(
        [(gene, centrality) for gene, centrality in grn.var["centrality"].items()],
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    print("Top central genes:", top_central_genes)
    return top_central_genes


def enrichment(
    grn,
    of="Targets",
    doplot=True,
    top_k=30,
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
    min_size=3,
    max_size=4000,
    permutation_num=1000,  # reduce number to speed up testing
    **kwargs
):
    """
    This function performs enrichment analysis on a given gene regulatory network (grn).

    Parameters:
        grn (GRNAnnData): The gene regulatory network to analyze.
        of (str, optional): The specific component of the grn to focus on.
        top_k (int, optional): The number of top results to return. Defaults to 10.
        doplot (bool, optional): Whether to generate a plot of the results. Defaults to False.

    Returns:
        pandas.DataFrame: A DataFrame containing the results of the enrichment analysis.
    """
    if of == "Targets":
        rnk = grn.grn.sum(1).sort_values(ascending=False)
    elif of == "Regulators":
        rnk = grn.grn.sum(0).sort_values(ascending=False)
    elif of == "Central":
        try:
            get_centrality(grn, top_k=0)
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
            permutation_num=permutation_num,
            **kwargs
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
        ax = dotplot(
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
    """ """
    enr = gp.enrich(
        gene_list=grn.var[
            (grn.get(target).varm[of] != 0)[0]
        ].index.tolist(),  # or gene_list=glist
        gene_sets=[
            "KEGG_2016",
            {"h.all": gmt},
            "ENCODE_TF_ChIP-seq_2014",
            "GO_Molecular_Function_2015",
            {"TFs": TF},
            "../../../GRnnData/grnndata/celltype.gmt",
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
    return enr2.results[enr2.results["Adjusted P-value"] < 0.2]


def similarity(grn, other_grn):
    """
    This function calculates the similarity between two gene regulatory networks (grns).

    Parameters:
        grn (GRNAnnData): The first gene regulatory network.
        other_grn (GRNAnnData): The second gene regulatory network.

    Returns:
        dict : A dictionary containing the similarity metrics between the two grns.
            {"spearman_corr", "precision", "precision_rand", "recall",
            "recall_rand", "accuracy", "accuracy_rand", "sim_expr",
            "intra_similarity_self", "intra_similarity_other"}
    """
    # similarity in expression
    selfX = grn.X
    otherX = other_grn.X
    if issparse(otherX):
        otherX = otherX.todense()
    otherXrand = otherX.copy()
    np.random.shuffle(otherXrand)
    if issparse(selfX):
        selfX = selfX.todense()
    selfXrand = selfX.copy()
    np.random.shuffle(selfXrand)
    # Compute intra matrix similarity
    intra_similarity_self = np.nan_to_num(
        np.diag(
            np.corrcoef(selfXrand.T, selfX.T)[
                selfXrand.shape[1] :, : selfXrand.shape[1]
            ]
        ),
        1,
    ).mean()
    intra_similarity_other = np.nan_to_num(
        np.diag(
            np.corrcoef(otherX.T, otherXrand.T)[otherX.shape[1] :, : otherX.shape[1]]
        ),
        1,
    ).mean()
    # Compute inter matrix similarity
    inter_similarity = np.nan_to_num(
        np.diag(np.corrcoef(selfX.T, otherX.T)[selfX.shape[1] :, : selfX.shape[1]]),
        1,
    ).mean()
    print("intra_similarity_self", intra_similarity_self)
    print("intra_similarity_other", intra_similarity_other)
    print("inter_similarity", inter_similarity)

    # similarity in network structure
    # Get the GRN network from both grn objects
    grn_self = grn.varp["GRN"]
    grn_other = other_grn.varp["GRN"]

    # Compute the number of similar edges
    # Similar edges are those where both are non-zero and have the same sign
    similar_edges = (
        (grn_self != 0) & (grn_other != 0) & (np.sign(grn_self) == np.sign(grn_other))
    )

    similar_edges_ct = np.sum(similar_edges)
    print("similar_edges_ct", similar_edges_ct)

    # Compute the total number of edges
    total_edges = np.sum(grn_self != 0)

    # Compute the total number of edges in the other GRN
    total_edges_other = np.sum((grn_other != 0))

    # Compute precision, recall, and accuracy
    precision = similar_edges_ct / total_edges
    recall = similar_edges_ct / total_edges_other
    accuracy = similar_edges_ct / (total_edges + total_edges_other - similar_edges_ct)
    print("precision", precision)
    print("recall", recall)
    print("accuracy", accuracy)
    # Compute the Spearman's rank correlation between the two overlapping sets of edges
    spearman_corr = scipy.stats.spearmanr(
        grn_self[similar_edges].flatten(), grn_other[similar_edges].flatten()
    )
    print("spearman_corr", spearman_corr)
    # Generate a random permutation of varp['grn'] matrix
    grn_other_rand = grn_other.copy()
    np.random.shuffle(grn_other_rand)

    # Recompute the number of similar edges
    similar_edges_rand = (
        (grn_self != 0)
        & (grn_other_rand != 0)
        & (np.sign(grn_self) == np.sign(grn_other_rand))
    )
    similar_edges_ct_rand = np.sum(similar_edges_rand)
    total_edges_rand = np.sum((grn_other_rand != 0))

    # Recompute precision, recall, and accuracy
    precision_rand = similar_edges_ct_rand / total_edges
    recall_rand = similar_edges_ct_rand / total_edges_rand
    accuracy_rand = similar_edges_ct_rand / (
        total_edges + total_edges_rand - similar_edges_ct_rand
    )
    print("rand precision", precision_rand)
    print("rand recall", recall_rand)
    print("rand accuracy", accuracy_rand)

    # compute graph edit distance

    # G1 = nx.from_numpy_array(grn_self)
    # G2 = nx.from_numpy_array(grn_other)
    # dist = nx.graph_edit_distance(
    #    G1,
    #    G2,
    #    node_match=None,
    #    edge_match=lambda e1, e2: np.sign(e1) == np.sign(e2),
    #    timeout=300,
    # )
    return {
        "spearman_corr": spearman_corr,
        "precision": precision,
        "precision_rand": precision_rand,
        "recall": recall,
        "recall_rand": recall_rand,
        "accuracy": accuracy,
        "accuracy_rand": accuracy_rand,
        "sim_expr": inter_similarity,
        "intra_similarity_self": intra_similarity_self,
        "intra_similarity_other": intra_similarity_other,
        # "dist": dist,
    }


def extract_main(grn):
    pass


def metrics(grn):
    """
    ### small worldness
    A small-world network is a type of mathematical graph in which most nodes are not neighbors of one another,
    but most nodes can be reached from every other by a small number of hops or steps.
    the sigma metric is a measure of small-worldness.
    if sigma > 1, the network is a small-world network.

    ### scale freeness
    A scale-free network is a network whose degree distribution follows a power law,
    at least asymptotically. The s metric is a measure of scale-freeness,
    defined as ( S(G)={\frac {s(G)}{s_{\max }}} ), where ( s_{\max } )
    is the maximum value of s(H) for H in the set of all graphs with degree distribution.
    A graph with small S(G) is "scale-rich," and a graph with S(G) close to 1 is "scale-free"

    Therefore, a graph is more scale-free when its S(G) value is closer to 1.
    The range of the s metric is between 0 and 1, where 0 indicates "scale-rich" and 1 indicates "scale-free"

    Parameters:
        grn : The Gene Regulatory Network for which the connectivities are to be computed.
        stretch : The maximum distance between nodes in the spanner compared to the original graph. Default is 2.

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
    fig2 = fit.plot_pdf(color="b", linewidth=2)
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


def compute_connectivities(grn, stretch=2):
    """
    This function computes the connectivities of a given Gene Regulatory Network (GRN).
    It uses the NetworkX library to convert the GRN into a graph and then computes the spanner of the graph.
    The spanner of a graph is a subgraph that approximates the original graph in terms of distances between nodes.
    The stretch parameter determines the maximum distance between nodes in the spanner compared to the original graph.
    The computed connectivities are then stored in the GRN object.

    Parameters:
        grn : The Gene Regulatory Network for which the connectivities are to be computed.
        stretch : The maximum distance between nodes in the spanner compared to the original graph. Default is 2.

    Returns:
    grn : The Gene Regulatory Network with the computed connectivities.
    """
    grn.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "params": {"method": "GRN"},
    }
    grn.varp["connectivities"] = nx.to_scipy_sparse_array(
        nx.spanner(nx.from_numpy_array(grn.varp["GRN"]), stretch=stretch)
    )
    return grn


def plot_cluster(grn, color=["louvain"], min_dist=0.5, spread=0.7, stretch=2, **kwargs):
    """
    This function plots the clusters of a given Gene Regulatory Network (GRN).
    It first computes the connectivities of the GRN and then performs Louvain clustering on the transpose of the GRN.
    The clusters are then visualized using UMAP.

    Parameters:
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
