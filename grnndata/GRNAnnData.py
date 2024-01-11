from anndata import AnnData
import scipy.sparse
import pandas as pd
import numpy as np


class GRNAnnData(AnnData):
    def __init__(self, *args, grn, **kwargs):
        """An AnnData object with a GRN matrix in varp["GRN"]

        Args:
        -----
            grn: scipy.sparse.csr_matrix | np.ndarray a matrix with zeros and non-zeros
                signifying the presence of an edge and the direction of the edge
                respectively. The matrix should be square and the rows and columns
                should correspond to the genes in the AnnData object.
                The row index correpond to genes that are regulators and the column
                index corresponds to genes that are targets.

        @see https://anndata.readthedocs.io for more informaiotn on AnnData objects
        """
        super().__init__(*args, **kwargs)
        self.varp["GRN"] = grn

    # add concat
    def concat(self, other):
        if not isinstance(other, GRNAnnData):
            raise ValueError("Can only concatenate with another GRNAnnData object")
        return GRNAnnData(
            self.concatenate(other),
            grn=scipy.sparse.vstack([self.varp["GRN"], other.varp["GRN"]]),
        )

    # add slice
    def __getitem__(self, *args):
        sub = super().__getitem__(*args)
        if sub.shape[1] < self.shape[1]:
            sub.varm["all_targets"] = self.varp["GRN"][
                self.var_names.isin(sub.var_names)
            ]
        return sub

    @property
    def grn(self):
        """
        Property that returns the gene regulatory network (GRN) as a pandas DataFrame.
        The index and columns of the DataFrame are the gene names stored in 'var_names'.

        Returns:
            pd.DataFrame: The GRN as a DataFrame with gene names as index and columns.
        """
        return pd.DataFrame(
            data=self.varp["GRN"], index=self.var_names, columns=self.var_names
        )

    # add return list of genes and corresponding weights
    def extract_links(
        self,
        columns=[
            "regulator",
            "target",
            "weight",
        ],  # output col names (e.g. 'TF', 'gene', 'score')
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
        return pd.DataFrame(
            [
                a
                for a in zip(
                    [self.var_names[i] for i in self.varp["GRN"].row],
                    [self.var_names[i] for i in self.varp["GRN"].col],
                    self.varp["GRN"].data,
                )
            ],
            columns=columns,
        ).sort_values(by=columns[2], ascending=False)

    @classmethod
    def read(cls, *args, **kwargs):
        return from_anndata(AnnData.read(*args, **kwargs))


def from_anndata(adata):
    if "GRN" not in adata.varp:
        raise ValueError("GRN not found in adata.varp")
    return GRNAnnData(adata, grn=adata.varp["GRN"])
