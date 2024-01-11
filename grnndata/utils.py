import networkx as nx
import gseapy as gp
from gseapy.plot import barplot, dotplot
import numpy as np
import os.path


def fileToList(filename, strconv=lambda x: x):
    """
    loads an input file with a\\n b\\n.. into a list [a,b,..]
    """
    with open(filename) as f:
        return [strconv(val[:-1]) for val in f.readlines()]


file_dir = os.path.dirname(os.path.realpath(__file__))
TF = fileToList(file_dir + "/TF.txt")


def get_centrality(grn, top_k=30):
    """
    get_centrality uses the networkx library to calculate the centrality of each node in the GRN.
    The centrality is added to the grn object as a new column in the var dataframe.
    also prints the top K most central nodes in the GRN.

    Args:
        grn (_type_): _description_
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


def enrichment(grn, of="Targets", doplot=True, top_k=30, **kwargs):
    """
    enrichment uses the gseapy library to calculate the enrichment of the target genes in the adata
    the enrichment is returned and plotted

    Args:
        grn (_type_): _description_
        of (str, optional): either ['Targets', 'Regulators', 'Central']. Defaults to "Targets".
        for_ (str, optional): _description_. Defaults to "TFs".
    """
    if of == "Targets":
        rnk = grn.grn.sum(1).sort_values(ascending=False)
    elif of == "Regulators":
        rnk = grn.grn.sum(0).sort_values(ascending=False)
    elif of == "Central":
        get_centrality(grn, top_k=0)
        rnk = grn.var["centrality"].sort_values(ascending=False)
    rnk.name = None
    # run enrichment analysis
    pre_res = gp.prerank(
        rnk=rnk,  # or rnk = rnk,
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
        min_size=5,
        max_size=1000,
        permutation_num=1000,  # reduce number to speed up testing
        outdir=None,  # don't write to disk
        seed=6,
        verbose=True,  # see what's going on behind the scenes
    )
    val = (
        pre_res.res2d[(pre_res.res2d["FDR q-val"] < 0.1) & (pre_res.res2d["NES"] > 1)]
        .sort_values(by=["NES"], ascending=False)
        .drop(columns=["Name"])
    )
    print(val.Term.tolist()[:top_k])
    # plot results
    if doplot:
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

    return val


def similarity(grn, other_grn):
    # similarity in expression
    selfX = grn.X
    selfXrand = selfX.copy()
    otherX = other_grn.X
    otherXrand = otherX.copy()
    np.random.shuffle(selfXrand.data)
    np.random.shuffle(otherXrand.data)
    # Compute intra matrix similarity
    intra_similarity_self = np.dot(selfX, selfXrand.T)
    intra_similarity_other = np.dot(otherX, otherXrand.T)
    # Compute inter matrix similarity
    inter_similarity = np.dot(selfX, otherX.T)

    # similarity in network structure
    # Get the GRN network from both grn objects
    grn_self = grn.varp["GRN"]
    grn_other = other_grn.varp["GRN"]

    # Compute the number of similar edges
    # Compute the number of similar edges
    # Similar edges are those where both are non-zero and have the same sign
    similar_edges = (
        (grn_self != 0) & (grn_other != 0) & (np.sign(grn_self) == np.sign(grn_other))
    )
    similar_edges_ct = np.sum(similar_edges)

    # Compute the total number of edges
    total_edges = np.sum(grn_self != 0)

    # Compute the total number of edges in the other GRN
    total_edges_other = np.sum((grn_other != 0))

    # Compute precision, recall, and accuracy
    precision = similar_edges_ct / total_edges
    recall = similar_edges_ct / total_edges_other
    accuracy = similar_edges_ct / (total_edges + total_edges_other - similar_edges_ct)

    # Compute the Spearman's rank correlation between the two overlapping sets of edges
    spearman_corr = scipy.stats.spearmanr(
        grn_self[similar_edges].flatten(), grn_other[similar_edges].flatten()
    )
    # Generate a random permutation of varp['grn'] matrix
    grn_self_rand = grn_self.copy()
    np.random.shuffle(grn_self_rand.data)

    # Recompute the number of similar edges
    similar_edges_rand = (
        (grn_self != 0)
        & (grn_self_rand != 0)
        & (np.sign(grn_self) == np.sign(grn_self_rand))
    )
    similar_edges_ct_rand = np.sum(similar_edges_rand)
    total_edges_rand = np.sum((grn_self_rand != 0))

    # Recompute precision, recall, and accuracy
    precision_rand = similar_edges_ct_rand / total_edges
    recall_rand = similar_edges_ct_rand / total_edges_rand
    accuracy_rand = similar_edges_ct_rand / (
        total_edges + total_edges_rand - similar_edges_ct_rand
    )
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
    }
