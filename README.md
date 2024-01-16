# grnndata

[![codecov](https://codecov.io/gh/cantinilab/GRnnData/branch/main/graph/badge.svg?token=GRnnData_token_here)](https://codecov.io/gh/cantinilab/GRnnData)
[![CI](https://github.com/cantinilab/GRnnData/actions/workflows/main.yml/badge.svg)](https://github.com/cantinilab/GRnnData/actions/workflows/main.yml)

Awesome gene regulatory network enhanced anndata created by @jkobject & remi trimbour

grnndata works similarly to anndata. The goal was to use the .varm of anndata to store the GRN data associated with a dataset and have a formal way to work with GRNs.

grnndata is a subclass of anndata.AnnData, it enforces only that a .varm exists for the anndata

grnndata also contains multiple helper functions to work with GRNs in scRNAseq

## Install it from PyPI

```bash
pip install grnndata
```

## Usage

```py
from grnndata import GRNAnnData
from grnndata import utils

grn = GRNAnnData(subdata.copy(), grn=grn[1:,1:])
utils.some_function(grn)
```

grnndata works similarly to anndata. Learn more about usages in the documentation and its notebooks in the [docs](https://cantinilab.github.io/GRnnData/). 

### How do I do if I generate a GRN per cell type?

In this context, we recommend creating a grnndata per cell type. This will allow you to store the GRN data in the .varm of the grnndata and have a formal way to work with GRNs.

### How do I do if I generate a GRN per cell?

In this context, we recommend trying to merge them across a similar group of cells in some way and storing uncertainty or variance in the GRN and then creating a grnndata across this group of cells

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.
