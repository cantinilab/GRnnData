from anndata import AnnData
import scipy.sparse
import pandas as pd
import numpy as np


class GRNAnnData(AnnData):
    def __init__(self, *args, grn, **kwargs):
        super().__init__(*args, **kwargs)

        self.varp["GRN"] = grn

    ## add concat
    def concat(self, other):
        if not isinstance(other, GRNAnnData):
            raise ValueError("Can only concatenate with another GRNAnnData object")
        return GRNAnnData(
            self.concatenate(other),
            grn=scipy.sparse.vstack([self.varp["GRN"], other.varp["GRN"]]),
        )

    ## add slice
    def __getitem__(self, name):
        if isinstance(name, str):
            index = self.var_names.tolist().index(name)
            return GRNAnnData(
                self[:, name],
                grn=self.varp["GRN"][index],
            )
        # need to put it in varm
        if isinstance(name, list):
            index = [self.var_names.tolist().index(i) for i in name]
            return GRNAnnData(grn=self.varp["GRN"][index], X=self.X[index])
        # need to put it in varm too
        if isinstance(name, np.ndarray):
            return GRNAnnData(grn=self.varp["GRN"][name], X=self.X[name])
        # need to put it in varm too
        if isinstance(name, slice):
            return GRNAnnData(
                grn=self.varp["GRN"][name, name],
                X=self.X[name],
            )
        # need to put it in varm too

    ## add return list of genes and corresponding weights
    def extract_links(
        adata,  # AnnData object
        columns=[
            "row",
            "col",
            "weight",
        ],  # output col names (e.g. 'TF', 'gene', 'score')
    ):
        """
        little function to extract scores from anndata.varp['key'] as a pd.DataFrame :
            TF   Gene   Score
            A        B          5
            C        D         8
        """
        return pd.DataFrame(
            [
                a
                for a in zip(
                    [adata.var_names[i] for i in adata.varp["GRN"].row],
                    [adata.var_names[i] for i in adata.varp["GRN"].col],
                    adata.varp["GRN"].data,
                )
            ],
            columns=columns,
        ).sort_values(by=columns[2], ascending=False)


def from_anndata(adata):
    if "GRN" not in adata.obsp:
        raise ValueError("GRN not found in adata.obsp")
    return GRNAnnData(adata, grn=adata.obsp["GRN"])
