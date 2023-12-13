from anndata import AnnData
import scipy

class GRNAnnData(AnnData):
    def __init__(self, grn, **kwargs):
        super().__init__(**kwargs)

        self.varp["GRN"] = grn

    ## add concat
    def concat(self, other):
        if not isinstance(other, GRNAnnData):
            raise ValueError("Can only concatenate with another GRNAnnData object")
        return GRNAnnData(
            grn = scipy.sparse.vstack([self.varp["GRN"], other.varp["GRN"]]),
            self.concatenate(other)
        )

    ## add slice
    def __getitem__(self, name):
        if isinstance(index, str):
            index = self.var_names.tolist().index(name)
        return GRNAnnData(
            grn = self.varp["GRN"][index],
            self[:,name]
        )
        #need to put it in varm
        if isinstance(name, list):
            index = [self.var_names.tolist().index(i) for i in name]
            return GRNAnnData(
                grn = self.varp["GRN"][index],
                X = self.X[index]
            )
        #need to put it in varm too
        if isinstance(index, np.ndarray):
            return GRNAnnData(
                grn = self.varp["GRN"][index],
                X = self.X[index]
            )
        #need to put it in varm too
        if isinstance(index, slice)
            return GRNAnnData(
                grn = self.varp["GRN"][index,index],
                X = self.X[index]
            )
        #need to put it in varm too

    ## add return list of genes and corresponding weights
    def extract_links(
        adata, #AnnData object
        columns = ['row', 'col', 'weight'] # output col names (e.g. 'TF', 'gene', 'score')
    ):
    """
    little function to extract scores from anndata.varp['key'] as a pd.DataFrame :
        TF   Gene   Score
        A        B          5
        C        D         8
    """
        return pd.DataFrame(
            [a for a in zip(
                [adata.var_names[i] for i in adata.varp['GRN'].row],
                [adata.var_names[i] for i in adata.varp['GRN'].col],
                adata.varp['GRN'].data)
            ],
            columns = columns
            )

def from_anndata(adata):
    if "GRN" not in adata.obsp:
        raise ValueError("GRN not found in adata.obsp")
    return GRNAnnData(adata.obsp["GRN"], X=adata)


def get_centrality(GRNAnnData, k=30):
    """
    get_centrality uses the networkx library to calculate the centrality of each node in the GRN.
    The centrality is added to the GRNAnnData object as a new column in the var dataframe.
    also prints the top K most central nodes in the GRN.

    Args:
        GRNAnnData (_type_): _description_
    """
    import networkx as nx

    G = nx.from_scipy_sparse_matrix(GRNAnnData.obsp["GRN"])
    centrality = nx.eigenvector_centrality(G)

    GRNAnnData.var["centrality"] = [
        centrality.get(gene, 0) for gene in GRNAnnData.var_names
    ]

    top_central_genes = sorted(
        [(node, centrality) for node, centrality in centrality.items()],
        key=lambda x: x[1],
        reverse=True,
    )[:k]
    print("Top central genes:", top_central_genes)


def enrichment(GRNAnnData, of="Targets", for_="TFs", doplot=True, **kwargs):
    """
    enrichment uses the gseapy library to calculate the enrichment of the target genes in the adata
    the enrichment is returned and plotted

    Args:
        GRNAnnData (_type_): _description_
        of (str, optional): _description_. Defaults to "Targets".
        for_ (str, optional): _description_. Defaults to "TFs".
    """
    import gseapy as gp
    from gseapy.plot import barplot, dotplot

    mapping = {
        "TFs": "KEGG_2019_Human",
    }

    # define gene sets
    if of == "Targets":
        gene_sets = GRNAnnData.var_names
    elif of == "TFs":
        gene_sets = GRNAnnData.var["TFs"]
    else:
        raise ValueError("of must be one of 'Targets', 'TFs'")

    # run enrichment analysis
    enr = gp.enrichr(
        gene_list=gene_sets, description=for_, gene_sets=mapping[for_], **kwargs
    )

    # plot results
    if doplot:
        barplot(enr.res2d, title=for_)

    return enr


def similarity(GRNAnnData, other_GRNAnnData):
    pass


def get_subnetwork(GRNAnnData, on="TFs"):
    if type(on) is list:
        pass
    elif on == "TFs":
        pass
    elif on == "Regulators":
        pass
    else:
        raise ValueError("on must be one of 'TFs', 'Regulators', or a list of genes")
    pass


def focuses_more_on(GRNAnnData, on="TFs"):
    pass
