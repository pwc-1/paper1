import scanpy as sc
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


def dopca(X, dim):
    pcaten = PCA(n_components=dim)
    X_10 = pcaten.fit_transform(X)
    return X_10


def read_dataset(File1=None, File2=None, File3=None, File4=None, transpose=True, test_size_prop=None, state=0,
                 format_rna=None, formar_epi=None):
    # read single-cell multi-omics data together

    ### raw reads count of scRNA-seq data
    adata = adata1 = None

    if File1 is not None:
        if format_rna == "table":
            adata = sc.read(File1)
        else:  # 10X format
            adata = sc.read_mtx(File1)

        if transpose:
            adata = adata.transpose()

    ##$ the binarization data for scEpigenomics file
    if File2 is not None:
        if formar_epi == "table":
            adata1 = sc.read(File2)
        else:  # 10X format
            adata1 = sc.read_mtx(File2)

        if transpose:
            adata1 = adata1.transpose()

    ### File3 and File4 for cell group information of scRNA-seq and scEpigenomics data
    label_ground_truth = []
    label_ground_truth1 = []

    if state == 0:
        if File3 is not None:
            Data2 = pd.read_csv(File3, header=0, index_col=0)
            label_ground_truth = Data2['Group'].values

        else:
            label_ground_truth = np.ones(len(adata.obs_names))

        if File4 is not None:
            Data2 = pd.read_csv(File4, header=0, index_col=0)
            label_ground_truth1 = Data2['Group'].values

        else:
            label_ground_truth1 = np.ones(len(adata.obs_names))

    elif state == 1:
        if File3 is not None:
            Data2 = pd.read_table(File3, header=0, index_col=0)
            label_ground_truth = Data2['cell_line'].values
        else:
            label_ground_truth = np.ones(len(adata.obs_names))

        if File4 is not None:
            Data2 = pd.read_table(File4, header=0, index_col=0)
            label_ground_truth1 = Data2['cell_line'].values
        else:
            label_ground_truth1 = np.ones(len(adata.obs_names))

    elif state == 3:
        if File3 is not None:
            Data2 = pd.read_table(File3, header=0, index_col=0)
            label_ground_truth = Data2['Group'].values
        else:
            label_ground_truth = np.ones(len(adata.obs_names))

        if File4 is not None:
            Data2 = pd.read_table(File4, header=0, index_col=0)
            label_ground_truth1 = Data2['Group'].values
        else:
            label_ground_truth1 = np.ones(len(adata.obs_names))

    else:
        if File3 is not None:
            Data2 = pd.read_table(File3, header=0, index_col=0)
            label_ground_truth = Data2['Cluster'].values
        else:
            label_ground_truth = np.ones(len(adata.obs_names))

        if File4 is not None:
            Data2 = pd.read_table(File4, header=0, index_col=0)
            label_ground_truth1 = Data2['Cluster'].values
        else:
            label_ground_truth1 = np.ones(len(adata.obs_names))

    # split datasets into training and testing sets
    if test_size_prop > 0:
        train_idx, test_idx = train_test_split(np.arange(adata.n_obs),
                                               test_size=test_size_prop,
                                               random_state=200)
        spl = pd.Series(['train'] * adata.n_obs)
        spl.iloc[test_idx] = 'test'
        adata.obs['split'] = spl.values

        if File2 is not None:
            adata1.obs['split'] = spl.values
    else:
        train_idx, test_idx = list(range(adata.n_obs)), list(range(adata.n_obs))
        spl = pd.Series(['train'] * adata.n_obs)
        adata.obs['split'] = spl.values

        if File2 is not None:
            adata1.obs['split'] = spl.values

    adata.obs['split'] = adata.obs['split'].astype('category')
    adata.obs['Group'] = label_ground_truth
    adata.obs['Group'] = adata.obs['Group'].astype('category')

    if File2 is not None:
        adata1.obs['split'] = adata1.obs['split'].astype('category')
        adata1.obs['Group'] = label_ground_truth
        adata1.obs['Group'] = adata1.obs['Group'].astype('category')

    # print('Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))

    ### here, adata with cells * features
    return adata, adata1, train_idx, test_idx, label_ground_truth, label_ground_truth1


def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=False, logtrans_input=True):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if logtrans_input:
        sc.pp.log1p(adata)

    if size_factors:
        adata.obs['size_factors'] = np.log(np.sum(adata.X, axis=1))
    else:
        adata.obs['size_factors'] = 1.0

    if normalize_input:
        sc.pp.scale(adata)

    return adata


def normalize2(adata, copy=True, highly_genes=None, filter_min_counts=True, size_factors=True, normalize_input=True,
               logtrans_input=True):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    if adata.X.size < 50e6:  # check if adata.X is integer only if array is small
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)
    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=highly_genes,
                                    subset=True)
    if normalize_input:
        sc.pp.scale(adata)
    return adata
