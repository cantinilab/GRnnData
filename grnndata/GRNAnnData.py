from typing import List, Optional

import gseapy as gp
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse
import seaborn as sns
import tqdm
from anndata import AnnData
from anndata import read_h5ad as anndata_read_h5ad
from d3graph import d3graph
from matplotlib import pyplot as plt
from pyvis import network as pnx
from sklearn.metrics.pairwise import cosine_similarity

# Get the base seaborn color palette as hex list
base_color_palette = sns.color_palette().as_hex()


class GRNAnnData(AnnData):
    def __init__(
        self,
        *args,
        grn: Optional[scipy.sparse.csr_matrix | np.ndarray] = None,
        **kwargs,
    ):
        """An AnnData object with a GRN matrix in varp["GRN"]

        Args:
            grn: scipy.sparse.csr_matrix | np.ndarray a matrix with zeros and non-zeros
                signifying the presence of an edge and the direction of the edge
                respectively. The matrix should be square and the rows and columns
                should correspond to the genes in the AnnData object.
                The row index correpond to genes that are regulators and the column
                index corresponds to genes that are targets.

        @see https://anndata.readthedocs.io for more informaiotn on AnnData objects
        """
        # if isinstance(args[0], AnnData) and "GRN" in args[0].varp:
        #    args[0] = args[0].copy()
        #     grn = args[0].varp["GRN"]
        # elif grn is None:
        #    raise ValueError("grn argument must be provided")
        super(GRNAnnData, self).__init__(*args, **kwargs)
        # Ensure GRN is stored in varp, not directly as an attribute
        if grn is not None:
            self.varp["GRN"] = grn
        # Ensure GRN key exists even if grn is None
        elif "GRN" not in self.varp:
            self.varp["GRN"] = None

    def copy(self, filename: str | None = None) -> "GRNAnnData":
        """Return a copy of the object."""
        adata_copy = super().copy(filename=filename)

        new_grn = None
        if "GRN" in self.varp and self.varp["GRN"] is not None:
            # Ensure the GRN matrix is copied as well
            new_grn = self.varp["GRN"].copy()

        return GRNAnnData(adata_copy, grn=new_grn)

    def __getitem__(self, index) -> "GRNAnnData":
        """Return a sliced view or copy of the object."""
        sliced_adata = super().__getitem__(index).copy()

        # If slicing resulted in something other than AnnData (e.g., a scalar), return it directly
        if not isinstance(sliced_adata, AnnData):
            return sliced_adata

        # If the original GRN doesn't exist, return the sliced AnnData as GRNAnnData without a GRN
        if "GRN" not in self.varp or self.varp["GRN"] is None:
            return GRNAnnData(sliced_adata, grn=None)

        original_grn = self.varp["GRN"]
        new_grn = None

        # Check if the variable dimension was sliced
        if sliced_adata.n_vars < self.n_vars:
            # Variables were sliced, subset the GRN accordingly
            try:
                # Find the integer indices of the remaining variables in the original object
                original_var_indices = self.var_names.get_indexer(
                    sliced_adata.var_names
                )

                if scipy.sparse.issparse(original_grn):
                    # Slice sparse matrix: select rows, then columns
                    new_grn = original_grn[original_var_indices][
                        :, original_var_indices
                    ]
                elif isinstance(original_grn, np.ndarray):
                    # Slice dense matrix using np.ix_ for correct indexing
                    new_grn = original_grn[
                        np.ix_(original_var_indices, original_var_indices)
                    ]
                else:
                    # Fallback or error if GRN type is unexpected
                    # For now, just pass None, but ideally handle or raise error
                    new_grn = None
            except Exception:
                # If slicing fails for any reason, maybe default to no GRN?
                # Or log a warning. For now, pass None.
                import warnings

                warnings.warn(
                    "Failed to slice GRN matrix. Resulting GRNAnnData will have grn=None."
                )
                new_grn = None

        elif sliced_adata.n_vars == self.n_vars:
            # Variable dimension not sliced, keep a copy of the original GRN
            new_grn = original_grn.copy()
        # Else case (n_vars > self.n_vars) shouldn't happen with standard slicing

        # Always return a GRNAnnData object
        # The sliced_adata might be a view, creating GRNAnnData might implicitly copy?
        # To be safe, explicitly copy the AnnData part if it's a view and we modified the GRN
        # However, AnnData's __getitem__ often returns a view for performance.
        # Let's stick to AnnData's default behavior regarding views/copies for the main data
        # and just handle the GRN part.
        return GRNAnnData(sliced_adata, grn=new_grn)

    # add concat
    def concat(self, other):
        """
        concat two GRNAnnData objects

        Args:
            other (GRNAnnData): The other GRNAnnData object to concatenate with

        Raises:
            ValueError: Can only concatenate with another GRNAnnData object

        Returns:
            AnnData: The concatenated GRNAnnData object
        """
        if not isinstance(other, GRNAnnData):
            raise ValueError("Can only concatenate with another GRNAnnData object")
        return GRNAnnData(
            self.concatenate(other),
            grn=scipy.sparse.vstack([self.varp["GRN"], other.varp["GRN"]]),
        )

    # add slice
    def get(self, elem: str | List[str]) -> "GRNAnnData":
        """
        get a sub-GRNAnnData object with only the specified genes

        Args:
            elem (str | list): The gene names to include in the sub-GRNAnnData object

        Returns:
            GRNAnnData: The sub-GRNAnnData object with only the specified genes
        """
        if type(elem) is str:
            elem = [elem]
        loc = self.var.index.isin(elem)
        reg = self.varp["GRN"][loc][:, loc]
        if len(reg.shape) == 1:
            reg = np.array([reg])
        sub = GRNAnnData(X=self.X[:, loc], obs=self.obs, var=self.var[loc], grn=reg)
        sub.varm["Targets"] = self.varp["GRN"][loc]
        sub.varm["Regulators"] = self.varp["GRN"].T[loc]
        sub.uns["regulated_genes"] = self.var.index.tolist()
        return sub

    @property
    def grn(self):
        """
        Property that returns the gene regulatory network (GRN) as a pandas DataFrame.
        The index and columns of the DataFrame are the gene names stored in 'var_names'.

        Returns:
            pd.DataFrame: The GRN as a DataFrame with gene names as index and columns.
        """
        if scipy.sparse.issparse(self.varp["GRN"]):
            data = self.varp["GRN"].toarray()
        else:
            data = self.varp["GRN"]
        return pd.DataFrame(data=data, index=self.var_names, columns=self.var_names)

    @property
    def regulators(self):
        """
        regulators outputs the regulators' connections of the GRN as a pandas DataFrame.

        Returns:
            pd.DataFrame: The regulators of the GRN as a DataFrame with gene names as index and columns.
        """
        if "Regulators" not in self.varm:
            return self.grn
        if scipy.sparse.issparse(self.varm["Regulators"]):
            data = self.varm["Regulators"].toarray()
        else:
            data = self.varm["Regulators"]
        return pd.DataFrame(
            data=data, index=self.var_names, columns=self.uns["regulated_genes"]
        )

    @property
    def targets(self):
        """
        targets outputs the targets' connections of the GRN as a pandas DataFrame.

        Returns:
            pd.DataFrame: The targets of the GRN as a DataFrame with gene names as index and columns.
        """
        if "Targets" not in self.varm:
            return self.grn.T
        if scipy.sparse.issparse(self.varm["Targets"]):
            data = self.varm["Targets"].toarray()
        else:
            data = self.varm["Targets"]
        return pd.DataFrame(
            data=data, index=self.var_names, columns=self.uns["regulated_genes"]
        )

    # add return list of genes and corresponding weights
    def extract_links(
        self,
    ):
        """
        This function extracts scores from anndata.varp['key'] and returns them as a pandas DataFrame.

        The resulting DataFrame has the following structure:
            TF   Gene   Score
            A    B      5
            C    D      8

        Where 'TF' and 'Gene' are the indices of the genes in the regulatory network, and 'Score' is the corresponding weight.

        Args:
            columns (list, optional): The names of the columns in the resulting DataFrame. Defaults to ['regulator', 'target', 'weight'].

        Returns:
            pd.DataFrame: The extracted scores as a DataFrame.
        """
        dfgrn = self.grn  # .columns.name
        dfgrn.columns.name = "targets"
        dfgrn.index.name = "regulators"
        dfgrn = dfgrn.reset_index()
        dfgrn = dfgrn.melt(
            id_vars="regulators", var_name="targets", value_name="weight"
        )
        dfgrn = dfgrn[dfgrn["weight"] != 0]
        return dfgrn

    def __repr__(self):
        text = super().__repr__()
        text += "\n    with a grn of {} elements".format((self.varp["GRN"] != 0).sum())
        return "GR" + text[1:]

    def plot_subgraph(
        self,
        seed: str,
        gene_col: str = "symbol",
        max_genes: int = 10,
        only: float = 0.3,
        palette: List[str] = base_color_palette,
        interactive: bool = True,
        do_enr: bool = False,
        color_overlap: pd.DataFrame | None = None,
        node_size: int = 3000,
        font_size: int = 14,
        **kwargs: dict,
    ):
        """
        plot_subgraph plots a subgraph of the gene regulatory network (GRN) centered around a seed gene.

        Args:
            seed (str or list): The seed gene or list of genes around which the subgraph will be centered.
            gene_col (str, optional): The column name in the .var DataFrame that contains gene identifiers. Defaults to "symbol".
            max_genes (int, optional): The maximum number of genes to include in the subgraph. Defaults to 10.
            only (float, optional): The threshold for filtering connections. If less than 1, it is used as a minimum weight threshold. If 1 or greater, it is used as the number of top connections to retain. Defaults to 0.3.
            palette (list, optional): The color palette to use for plotting. Defaults to base_color_palette.
            interactive (bool, optional): Whether to create an interactive plot. Defaults to True.
            do_enr (bool, optional): Whether to perform enrichment analysis on the subgraph. Defaults to False.
            color_overlap (pd.DataFrame | None, optional): A DataFrame with geneA, geneB, and value columns to color edges based on overlap. Defaults to None.

        Returns:
            d3graph or None: The d3graph object if interactive is True, otherwise None.
        """
        rn = {k: v for k, v in self.var[gene_col].items()}
        if type(seed) is str:
            gene_id = self.var[self.var[gene_col] == seed].index[0]
            elem = (
                self.grn.loc[gene_id]
                .sort_values(ascending=False)
                .head(max_genes)
                .index.tolist()
            )
            if gene_id not in elem:
                elem += [gene_id]
        else:
            elem = seed
        mat = self.grn.loc[elem, elem].rename(columns=rn).rename(index=rn)
        if only < 1:
            mat[mat < only] = 0
        else:
            top_connections = mat.stack().nlargest(only)
            top_connections.index.names = ["Gene1", "Gene2"]
            top_connections.name = "Weight"
            top_connections = top_connections.reset_index()
            mat.index.name += "_2"
            # Set anything not in the top N connections to 0
            mask = mat.stack().isin(
                top_connections.set_index(["Gene1", "Gene2"])["Weight"]
            )
            mat[~mask.unstack()] = 0
        mat = mat * 100
        color = [palette[0]] * len(mat)
        if type(seed) is str:
            color[mat.columns.get_loc(seed)] = palette[1]
        mat = mat.T

        if color_overlap is not None:
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors

            # Create a mapping from (geneA, geneB) to value
            edge_colors = {}
            for _, row in color_overlap.iterrows():
                edge_colors[(row["geneA"], row["geneB"])] = row["value"]

            # Get min and max values for normalization
            values = color_overlap["value"].values
            vmin, vmax = values.min(), values.max()

            # Create viridis colormap and normalizer
            cmap = cm.get_cmap("viridis")
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # print(mat)
        # print(color)
        if interactive:
            d3 = d3graph()
            d3.graph(mat, color=None)
            d3.set_node_properties(color=color, fontcolor="#000000", **kwargs)
            d3.set_edge_properties(directed=True)
            for i, gene1 in enumerate(mat.index):
                for j, gene2 in enumerate(mat.columns):
                    if mat.iloc[i, j] != 0 and gene1 != gene2:
                        if (gene1, gene2) in edge_colors:
                            rgba = cmap(norm(edge_colors[(gene1, gene2)]))
                            hex_color = mcolors.rgb2hex(rgba)
                            d3.edge_properties[(gene1, gene2)]["color"] = hex_color
                        else:
                            d3.edge_properties[(gene1, gene2)][
                                "color"
                            ] = "#808080"  # Default gray
            d3.show(notebook=True)
            return d3
        else:
            # Create a graph from the DataFrame
            G = nx.from_pandas_adjacency(mat, create_using=nx.DiGraph())
            # Draw the graph
            plt.figure(figsize=(15, 15))  # Increase the size of the plot
            # Use spring layout with adjusted parameters for better spacing
            pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)

            # Or try kamada_kawai layout for better node distribution
            # pos = nx.kamada_kawai_layout(G)

            # Draw nodes with larger size
            nx.draw_networkx_nodes(
                G, pos, node_size=node_size, node_color=color, alpha=0.6
            )

            if color_overlap is not None:
                # For networkx plotting
                edge_colors_nx = []
                for edge in G.edges():
                    if edge in edge_colors:
                        edge_colors_nx.append(cmap(norm(edge_colors[edge])))
                    elif (edge[1], edge[0]) in edge_colors:  # Check reverse direction
                        edge_colors_nx.append(
                            cmap(norm(edge_colors[(edge[1], edge[0])]))
                        )
                    else:
                        edge_colors_nx.append("#808080")  # Default gray
                # Draw edges with transparency
                nx.draw_networkx_edges(
                    G,
                    pos,
                    alpha=0.7,
                    width=1.5,
                    edge_color=edge_colors_nx,
                    arrows=True,
                    edge_cmap=cmap,
                    node_size=node_size,
                    connectionstyle="arc3,rad=0.1",
                    min_source_margin=15,
                    min_target_margin=15,
                )
                # Add colorbar
                sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                plt.colorbar(sm, ax=plt.gca(), label="Overlap Value")
            else:
                nx.draw_networkx_edges(
                    G,
                    pos,
                    alpha=0.7,
                    width=1.5,
                    arrows=True,
                    node_size=node_size,
                    connectionstyle="arc3,rad=0.1",
                    min_source_margin=15,
                    min_target_margin=15,
                )
            # Draw labels with better formatting
            nx.draw_networkx_labels(
                G,
                pos,
                font_size=font_size,
                font_weight="bold",
                font_family="sans-serif",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.7,
                ),
            )
            plt.axis("off")
            plt.tight_layout()
            plt.show()
        if do_enr:
            enr = gp.enrichr(
                gene_list=list(G.nodes),
                gene_sets=[
                    "KEGG_2021_Human",
                    "MSigDB_Hallmark_2020",
                    "Reactome_2022",
                    "Tabula_Sapiens",
                    "WikiPathway_2023_Human",
                    "TF_Perturbations_Followed_by_Expression",
                    "Reactome",
                    "PPI_Hub_Proteins",
                    "OMIM_Disease",
                    "GO_Molecular_Function_2023",
                ],
                organism="Human",  # change accordingly
                # description='pathway',
                # cutoff=0.08, # test dataset, use lower value for real case
                background=self.var.symbol.tolist(),
            )
            print(enr.res2d.head(20))
        return G


def read_h5ad(*args, **kwargs):
    """same as anndata's one but for grnndata"""
    return from_anndata(anndata_read_h5ad(*args, **kwargs))


def from_anndata(adata):
    """converts an anndata with a varp['GRN'] to a GRNAnnData"""
    if "GRN" not in adata.varp:
        raise ValueError("GRN not found in adata.varp")
    adata._X = adata.X
    return GRNAnnData(adata, grn=adata.varp["GRN"])


def from_scope_loomfile(filepath):
    """
    from_scope_loomfile creates a GRNAnnData object from a SCope Loom file.

    Args:
        filepath (str): The path to the SCope Loom file.

    Returns:
        GRNAnnData: A GRNAnnData object created from the SCope Loom file.
    """
    from loomxpy.loomxpy import SCopeLoom

    scopefile = SCopeLoom.read_loom(filepath)
    adata = AnnData(
        scopefile.ex_mtx,
        obs=pd.concat(
            [pd.DataFrame(v, columns=[k]) for k, v in scopefile.col_attrs.items()],
            axis=1,
        ).set_index("CellID"),
        var=pd.concat(
            [pd.DataFrame(v, columns=[k]) for k, v in scopefile.row_attrs.items()],
            axis=1,
        ).set_index("Gene"),
    )
    for k, v in scopefile.embeddings.items():
        adata.obsm[k] = v.embedding.values

    adata.uns = {
        i["name"]: i["values"]
        for i in scopefile.global_attrs["MetaData"]["annotations"]
    }
    adata.uns["regulon_thresholds"] = scopefile.global_attrs["MetaData"][
        "regulonThresholds"
    ]

    regulons_array = np.asarray(scopefile.row_attrs["Regulons"])
    regulons_df = pd.DataFrame(
        [reg.tolist() for reg in regulons_array], columns=regulons_array.dtype.names
    )
    regulons_df.index = adata.var.index
    adata.varm["regulons"] = regulons_df

    varnames = adata.var_names.tolist()
    da = np.zeros((len(varnames), len(varnames)), dtype=float)
    for i, (_, row) in enumerate(adata.varm["regulons"].iterrows()):
        names = row[row > 0].index.tolist()
        for name in names:
            sign = name.split("_")[1]
            name = name.split("_")[0]
            da[i, varnames.index(name)] = 1 if sign == "+" else -1
    return GRNAnnData(adata, grn=da)


def from_adata_and_longform(
    adata: AnnData, longform_df: pd.DataFrame, has_weight: bool = False
) -> GRNAnnData:
    """
    from_adata_and_longform creates a GRNAnnData object from an AnnData object and a longform DataFrame.

    the longform DataFrame should have the following structure:
        regulator  target  weight
        gene1      gene2  0.5
        gene2      gene3  0.7

    Args:
        adata (AnnData): An AnnData object containing gene expression data.
        longform_df (pd.DataFrame): A DataFrame in longform format with columns 'regulator', 'target', and optionally 'weight'.
        has_weight (bool, optional): If True, the 'weight' column in longform_df is used to set edge weights. Defaults to False.

    Returns:
        GRNAnnData: A GRNAnnData object containing the gene regulatory network.
    """
    varnames = adata.var_names.tolist()
    da = np.zeros((len(varnames), len(varnames)), dtype=float)
    svar = set(varnames)
    if has_weight:
        for i, j, v in tqdm.tqdm(longform_df.values):
            if i in svar and j in svar:
                da[varnames.index(i), varnames.index(j)] = v
    else:
        for i, j in tqdm.tqdm(longform_df.values):
            if i in svar and j in svar:
                da[varnames.index(i), varnames.index(j)] = 1
    return GRNAnnData(adata, grn=da)


def from_embeddings(
    adata: AnnData, layers: str = "emb", threshold: float = 0.4
) -> GRNAnnData:
    """
    from_embeddings creates a GRNAnnData object from an AnnData object with embeddings in one of its .varm[]s.

    Args:
        adata (AnnData): An AnnData object containing gene expression data and embeddings.
        layers (str, optional): The key in adata.varm where the embeddings are stored. Defaults to 'emb'.
        threshold (float, optional): The similarity threshold below which connections are discarded. Defaults to 0.4.

    Returns:
        GRNAnnData: A GRNAnnData object containing the gene regulatory network derived from the embeddings.
    """
    a = adata.varm[layers]
    similarities = cosine_similarity(a)
    similarities[similarities < 0.4] = 0
    embeddings_gnndata = GRNAnnData(adata, grn=scipy.sparse.coo_array(similarities))
    return embeddings_gnndata
