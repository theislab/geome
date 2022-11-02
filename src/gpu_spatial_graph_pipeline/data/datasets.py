"""
Copy pasted from https://github.com/theislab/ncem/blob/main/ncem/data.py
But renamed the DatasetX class to Dataset to avoid confusion with the pytorch Datasets.
"""

import abc
import warnings
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import squidpy as sq
from anndata import AnnData, read_h5ad
from pandas import read_csv, read_excel, DataFrame
from scipy import sparse


class GraphTools:
    """GraphTools class."""

    celldata: AnnData
    img_celldata: Dict[str, AnnData]

    def compute_adjacency_matrices(
        self,
        radius: int,
        coord_type: str = "generic",
        n_rings: int = 1,
        transform: str = None,
    ):
        """Compute adjacency matrix for each image in dataset (uses `squidpy.gr.spatial_neighbors`).

        Parameters
        ----------
        radius : int
            Radius of neighbors for non-grid data.
        coord_type : str
            Type of coordinate system.
        n_rings : int
            Number of rings of neighbors for grid data.
        transform : str
            Type of adjacency matrix transform. Valid options are:

            - `spectral` - spectral transformation of the adjacency matrix.
            - `cosine` - cosine transformation of the adjacency matrix.
            - `None` - no transformation of the adjacency matrix.
        """
        for _k, adata in self.img_celldata.items():
            if coord_type == "grid":
                radius = None
            else:
                n_rings = 1
            sq.gr.spatial_neighbors(
                adata=adata,
                coord_type=coord_type,
                radius=radius,
                n_rings=n_rings,
                transform=transform,
                key_added="adjacency_matrix",
            )

    @staticmethod
    def _transform_a(a):
        """Compute degree transformation of adjacency matrix.

        Computes D^(-1) * (A+I), with A an adjacency matrix, I the identity matrix and D the degree matrix.

        Parameters
        ----------
        a
            sparse adjacency matrix.

        Returns
        -------
        degree transformed sparse adjacency matrix
        """
        warnings.filterwarnings(
            "ignore", message="divide by zero encountered in true_divide"
        )
        degrees = 1 / a.sum(axis=0)
        degrees[a.sum(axis=0) == 0] = 0
        degrees = np.squeeze(np.asarray(degrees))
        deg_matrix = sparse.diags(degrees)
        a_out = deg_matrix * a
        return a_out

    def _transform_all_a(self, a_dict: dict):
        """Compute degree transformation for dictionary of adjacency matrices.

        Computes D^(-1) * (A+I), with A an adjacency matrix, I the identity matrix and D the degree matrix for all
        matrices in a dictionary.

        Parameters
        ----------
        a_dict : dict
            a_dict

        Returns
        -------
        dictionary of degree transformed sparse adjacency matrices
        """
        a_transformed = {i: self._transform_a(a) for i, a in a_dict.items()}
        return a_transformed

    @staticmethod
    def _compute_distance_matrix(pos_matrix):
        """Compute distance matrix.

        Parameters
        ----------
        pos_matrix
            Position matrix.

        Returns
        -------
        distance matrix
        """
        diff = pos_matrix[:, :, None] - pos_matrix[:, :, None].T
        return (diff * diff).sum(1)

    def _get_degrees(self, max_distances: list):
        """Get dgrees.

        Parameters
        ----------
        max_distances : list
            List of maximal distances.

        Returns
        -------
        degrees
        """
        degs = {}
        degrees = {}
        for i, adata in self.img_celldata.items():
            pm = np.array(adata.obsm["spatial"])
            dist_matrix = self._compute_distance_matrix(pm)
            degs[i] = {
                dist: np.sum(dist_matrix < dist * dist, axis=0)
                for dist in max_distances
            }
        for dist in max_distances:
            degrees[dist] = [deg[dist] for deg in degs.values()]
        return degrees


class Dataset(GraphTools):
    """Dataset class. Inherits all functions from GraphTools."""

    def __init__(
        self,
        data_path: str,
        radius: Optional[int] = None,
        coord_type: str = "generic",
        n_rings: int = 1,
        label_selection: Optional[List[str]] = None,
        n_top_genes: Optional[int] = None,
    ):
        """Initialize Dataset.

        Parameters
        ----------
        data_path : str
            Data path.
        radius : int
            Radius.
        label_selection : list, optional
            label selection.
        """
        self.data_path = data_path

        print("Loading data from raw files")
        self.register_celldata(n_top_genes=n_top_genes)
        self.register_img_celldata()
        self.register_graph_features(label_selection=label_selection)
        self.compute_adjacency_matrices(
            radius=radius, coord_type=coord_type, n_rings=n_rings
        )
        self.radius = radius

        print(
            "Loaded %i images with complete data from %i patients "
            "over %i cells with %i cell features and %i distinct celltypes."
            % (
                len(self.img_celldata),
                len(self.patients),
                self.celldata.shape[0],
                self.celldata.shape[1],
                len(self.celldata.uns["node_type_names"]),
            )
        )

    @property
    def patients(self):
        """Return number of patients in celldata.

        Returns
        -------
        patients
        """
        return np.unique(
            np.asarray(list(self.celldata.uns["img_to_patient_dict"].values()))
        )

    def register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""
        print("registering celldata")
        self._register_celldata(n_top_genes=n_top_genes)
        assert self.celldata is not None, "celldata was not loaded"

    def register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {imgage key : anndata object of image}."""
        print("collecting image-wise celldata")
        self._register_img_celldata()
        assert self.img_celldata is not None, "image-wise celldata was not loaded"

    def register_graph_features(self, label_selection):
        """Load graph level covariates.

        Parameters
        ----------
        label_selection
            Label selection.
        """
        print("adding graph-level covariates")
        self._register_graph_features(label_selection=label_selection)

    @abc.abstractmethod
    def _register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""
        pass

    @abc.abstractmethod
    def _register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {imgage key : anndata object of image}."""
        pass

    @abc.abstractmethod
    def _register_graph_features(self, label_selection):
        """Load graph level covariates.

        Parameters
        ----------
        label_selection
            Label selection.
        """
        pass

    def size_factors(self):
        """Get size factors (Only makes sense with positive input).

        Returns
        -------
        sf_dict
        """
        # Check if irregular sums are encountered:
        for i, adata in self.img_celldata.items():
            if np.any(np.sum(adata.X, axis=1) <= 0):
                print("WARNING: found irregular node sizes in image %s" % str(i))
        # Get global mean of feature intensity across all features:
        global_mean_per_node = self.celldata.X.sum(axis=1).mean(axis=0)
        return {
            i: global_mean_per_node / np.sum(adata.X, axis=1)
            for i, adata in self.img_celldata.items()
        }

    @property
    def var_names(self):
        return self.celldata.var_names


class DatasetZhang(Dataset):
    """DatasetZhang class. Inherits all functions from Dataset."""

    cell_type_merge_dict = {
        "Astrocytes": "Astrocytes",
        "Endothelial": "Endothelial",
        "L23_IT": "L2/3 IT",
        "L45_IT": "L4/5 IT",
        "L5_IT": "L5 IT",
        "L5_PT": "L5 PT",
        "L56_NP": "L5/6 NP",
        "L6_CT": "L6 CT",
        "L6_IT": "L6 IT",
        "L6_IT_Car3": "L6 IT Car3",
        "L6b": "L6b",
        "Lamp5": "Lamp5",
        "Microglia": "Microglia",
        "OPC": "OPC",
        "Oligodendrocytes": "Oligodendrocytes",
        "PVM": "PVM",
        "Pericytes": "Pericytes",
        "Pvalb": "Pvalb",
        "SMC": "SMC",
        "Sncg": "Sncg",
        "Sst": "Sst",
        "Sst_Chodl": "Sst Chodl",
        "VLMC": "VLMC",
        "Vip": "Vip",
        "other": "other",
    }

    def _register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""
        metadata = {
            "lateral_resolution": 0.109,
            "fn": "preprocessed_zhang.h5ad",
            "image_col": "slice_id",
            "pos_cols": ["center_x", "center_y"],
            "cluster_col": "subclass",
            "cluster_col_preprocessed": "subclass_preprocessed",
            "patient_col": "mouse",
        }
        celldata = read_h5ad(self.data_path + metadata["fn"]).copy()

        celldata.uns["metadata"] = metadata
        celldata.uns["img_keys"] = list(np.unique(celldata.obs[metadata["image_col"]]))

        img_to_patient_dict = {
            str(x): celldata.obs[metadata["patient_col"]].values[i].split("_")[0]
            for i, x in enumerate(celldata.obs[metadata["image_col"]].values)
        }

        celldata.uns["img_to_patient_dict"] = img_to_patient_dict
        self.img_to_patient_dict = img_to_patient_dict

        # register x and y coordinates into obsm
        celldata.obsm["spatial"] = celldata.obs[metadata["pos_cols"]]

        # add clean cluster column which removes regular expression from cluster_col
        celldata.obs[metadata["cluster_col_preprocessed"]] = list(
            pd.Series(list(celldata.obs[metadata["cluster_col"]]), dtype="str").map(
                self.cell_type_merge_dict
            )
        )
        celldata.obs[metadata["cluster_col_preprocessed"]] = celldata.obs[
            metadata["cluster_col_preprocessed"]
        ].astype("category")

        # register node type names
        node_type_names = list(
            np.unique(celldata.obs[metadata["cluster_col_preprocessed"]])
        )
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [
                node_type_names.index(x)
                for x in celldata.obs[metadata["cluster_col_preprocessed"]].values
            ]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def _register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {imgage key : anndata object of image}."""
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[str(k)] = self.celldata[
                self.celldata.obs[image_col] == k
            ].copy()
        self.img_celldata = img_celldata

    def _register_graph_features(self, label_selection):
        """Load graph level covariates.

        Parameters
        ----------
        label_selection
            Label selection.
        """
        # Save processed data to attributes.
        for adata in self.img_celldata.values():
            graph_covariates = {
                "label_names": {},
                "label_tensors": {},
                "label_selection": [],
                "continuous_mean": {},
                "continuous_std": {},
                "label_data_types": {},
            }
            adata.uns["graph_covariates"] = graph_covariates

        graph_covariates = {
            "label_names": {},
            "label_selection": [],
            "continuous_mean": {},
            "continuous_std": {},
            "label_data_types": {},
        }
        self.celldata.uns["graph_covariates"] = graph_covariates


class DatasetJarosch(Dataset):
    """DatasetJarosch class. Inherits all functions from Dataset."""

    cell_type_merge_dict = {
        "B cells": "B cells",
        "CD4 T cells": "CD4 T cells",
        "CD8 T cells": "CD8 T cells",
        "GATA3+ epithelial": "GATA3+ epithelial",
        "Ki67 high epithelial": "Ki67 epithelial",
        "Ki67 low epithelial": "Ki67 epithelial",
        "Lamina propria cells": "Lamina propria cells",
        "Macrophages": "Macrophages",
        "Monocytes": "Monocytes",
        "PD-L1+ cells": "PD-L1+ cells",
        "intraepithelial Lymphocytes": "intraepithelial Lymphocytes",
        "muscular cells": "muscular cells",
        "other Lymphocytes": "other Lymphocytes",
    }

    def _register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""
        metadata = {
            "lateral_resolution": 0.5,
            "fn": "raw_inflamed_colon_1.h5ad",
            "image_col": "Annotation",
            "pos_cols": ["X", "Y"],
            "cluster_col": "celltype_Level_2",
            "cluster_col_preprocessed": "celltype_Level_2_preprocessed",
            "patient_col": None,
        }
        celldata = read_h5ad(os.path.join(self.data_path, metadata["fn"]))
        feature_cols_hgnc_names = [
            "CD14",
            "MS4A1",
            "IL2RA",
            "CD3G",
            "CD4",
            "PTPRC",
            "PTPRC",
            "PTPRC",
            "CD68",
            "CD8A",
            "KRT5",  # 'KRT1', 'KRT14'
            "FOXP3",
            "GATA3",
            "MKI67",
            "Nuclei",
            "PDCD1",
            "CD274",
            "SMN1",
            "VIM",
        ]
        X = DataFrame(celldata.X, columns=feature_cols_hgnc_names)
        celldata = AnnData(
            X=X,
            obs=celldata.obs,
            uns=celldata.uns,
            obsm=celldata.obsm,
            varm=celldata.varm,
            obsp=celldata.obsp,
        )
        celldata.var_names_make_unique()
        celldata = celldata[celldata.obs[metadata["image_col"]] != "Dirt"].copy()
        celldata.uns["metadata"] = metadata
        img_keys = list(np.unique(celldata.obs[metadata["image_col"]]))
        celldata.uns["img_keys"] = img_keys

        # register x and y coordinates into obsm
        celldata.obsm["spatial"] = celldata.obs[metadata["pos_cols"]]

        img_to_patient_dict = {k: "p_1" for k in img_keys}
        celldata.uns["img_to_patient_dict"] = img_to_patient_dict
        self.img_to_patient_dict = img_to_patient_dict

        # add clean cluster column which removes regular expression from cluster_col
        celldata.obs[metadata["cluster_col_preprocessed"]] = list(
            pd.Series(list(celldata.obs[metadata["cluster_col"]]), dtype="str").map(
                self.cell_type_merge_dict
            )
        )
        celldata.obs[metadata["cluster_col_preprocessed"]] = celldata.obs[
            metadata["cluster_col_preprocessed"]
        ].astype("category")

        # register node type names
        node_type_names = list(
            np.unique(celldata.obs[metadata["cluster_col_preprocessed"]])
        )
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [
                node_type_names.index(x)
                for x in celldata.obs[metadata["cluster_col_preprocessed"]].values
            ]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def _register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {imgage key : anndata object of image}."""
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[str(k)] = self.celldata[
                self.celldata.obs[image_col] == k
            ].copy()
        self.img_celldata = img_celldata

    def _register_graph_features(self, label_selection):
        """Load graph level covariates.

        Parameters
        ----------
        label_selection
            Label selection.
        """
        # Save processed data to attributes.
        for adata in self.img_celldata.values():
            graph_covariates = {
                "label_names": {},
                "label_tensors": {},
                "label_selection": [],
                "continuous_mean": {},
                "continuous_std": {},
                "label_data_types": {},
            }
            adata.uns["graph_covariates"] = graph_covariates

        graph_covariates = {
            "label_names": {},
            "label_selection": [],
            "continuous_mean": {},
            "continuous_std": {},
            "label_data_types": {},
        }
        self.celldata.uns["graph_covariates"] = graph_covariates


class DatasetHartmann(Dataset):
    """DatasetHartmann class. Inherits all functions from Dataset."""

    cell_type_merge_dict = {
        "Imm_other": "Other immune cells",
        "Epithelial": "Epithelial",
        "Tcell_CD4": "CD4 T cells",
        "Myeloid_CD68": "CD68 Myeloid",
        "Fibroblast": "Fibroblast",
        "Tcell_CD8": "CD8 T cells",
        "Endothelial": "Endothelial",
        "Myeloid_CD11c": "CD11c Myeloid",
    }

    def _register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""
        metadata = {
            "lateral_resolution": 400 / 1024,
            "fn": [
                "scMEP_MIBI_singlecell/scMEP_MIBI_singlecell.csv",
                "scMEP_sample_description.xlsx",
            ],
            "image_col": "point",
            "pos_cols": ["center_colcoord", "center_rowcoord"],
            "cluster_col": "Cluster",
            "cluster_col_preprocessed": "Cluster_preprocessed",
            "patient_col": "donor",
        }
        celldata_df = read_csv(os.path.join(self.data_path, metadata["fn"][0]))
        celldata_df["point"] = [f"scMEP_point_{str(x)}" for x in celldata_df["point"]]
        celldata_df = celldata_df.fillna(0)
        # celldata_df = celldata_df.dropna(inplace=False).reset_index()
        feature_cols = [
            "H3",
            "vimentin",
            "SMA",
            "CD98",
            "NRF2p",
            "CD4",
            "CD14",
            "CD45",
            "PD1",
            "CD31",
            "SDHA",
            "Ki67",
            "CS",
            "S6p",
            "CD11c",
            "CD68",
            "CD36",
            "ATP5A",
            "CD3",
            "CD39",
            "VDAC1",
            "G6PD",
            "XBP1",
            "PKM2",
            "ASCT2",
            "GLUT1",
            "CD8",
            "CD57",
            "LDHA",
            "IDH2",
            "HK1",
            "Ecad",
            "CPT1A",
            "CK",
            "NaKATPase",
            "HIF1A",
            # "X1",
            # "cell_size",
            # "category",
            # "donor",
            # "Cluster",
        ]
        var_names = [
            "H3-4",
            "VIM",
            "SMN1",
            "SLC3A2",
            "NFE2L2",
            "CD4",
            "CD14",
            "PTPRC",
            "PDCD1",
            "PECAM1",
            "SDHA",
            "MKI67",
            "CS",
            "RPS6",
            "ITGAX",
            "CD68",
            "CD36",
            "ATP5F1A",
            "CD247",
            "ENTPD1",
            "VDAC1",
            "G6PD",
            "XBP1",
            "PKM",
            "SLC1A5",
            "SLC2A1",
            "CD8A",
            "B3GAT1",
            "LDHA",
            "IDH2",
            "HK1",
            "CDH1",
            "CPT1A",
            "CKM",
            "ATP1A1",
            "HIF1A",
        ]
        X = pd.DataFrame(np.array(celldata_df[feature_cols]), columns=var_names)
        celldata = AnnData(
            X=X,
            obs=celldata_df[
                ["point", "cell_id", "cell_size", "donor", "Cluster"]
            ].astype("category"),
            dtype=X.dtypes,
        )

        celldata.uns["metadata"] = metadata
        img_keys = list(np.unique(celldata_df[metadata["image_col"]]))
        celldata.uns["img_keys"] = img_keys

        # register x and y coordinates into obsm
        celldata.obsm["spatial"] = np.array(celldata_df[metadata["pos_cols"]])

        img_to_patient_dict = {
            str(x): celldata_df[metadata["patient_col"]].values[i]
            for i, x in enumerate(celldata_df[metadata["image_col"]].values)
        }
        # img_to_patient_dict = {k: "p_1" for k in img_keys}
        celldata.uns["img_to_patient_dict"] = img_to_patient_dict
        self.img_to_patient_dict = img_to_patient_dict

        # add clean cluster column which removes regular expression from cluster_col
        celldata.obs[metadata["cluster_col_preprocessed"]] = list(
            pd.Series(
                list(celldata.obs[metadata["cluster_col"]]), dtype="category"
            ).map(self.cell_type_merge_dict)
        )
        celldata.obs[metadata["cluster_col_preprocessed"]] = celldata.obs[
            metadata["cluster_col_preprocessed"]
        ].astype("category")

        # register node type names
        node_type_names = list(
            np.unique(celldata.obs[metadata["cluster_col_preprocessed"]])
        )
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [
                node_type_names.index(x)
                for x in celldata.obs[metadata["cluster_col_preprocessed"]].values
            ]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def _register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {imgage key : anndata object of image}."""
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[str(k)] = self.celldata[
                self.celldata.obs[image_col] == k
            ].copy()
        self.img_celldata = img_celldata

    def _register_graph_features(self, label_selection):
        """Load graph level covariates.

        Parameters
        ----------
        label_selection
            Label selection.
        """
        # DEFINE COLUMN NAMES FOR TABULAR DATA.
        # Define column names to extract from patient-wise tabular data:
        patient_col = "ID"
        # These are required to assign the image to dieased and non-diseased:
        disease_features = {"Diagnosis": "categorical"}
        patient_features = {
            "ID": "categorical",
            "Age": "continuous",
            "Sex": "categorical",
        }
        label_cols = {}
        label_cols.update(disease_features)
        label_cols.update(patient_features)

        if label_selection is None:
            label_selection = set(label_cols.keys())
        else:
            label_selection = set(label_selection)
        label_cols_toread = list(
            label_selection.intersection(set(list(label_cols.keys())))
        )
        usecols = label_cols_toread + [patient_col]

        tissue_meta_data = read_excel(
            os.path.join(self.data_path, "scMEP_sample_description.xlsx"),
            usecols=usecols,
        )
        # BUILD LABEL VECTORS FROM LABEL COLUMNS
        # The columns contain unprocessed numeric and categorical entries that are now processed to prediction-ready
        # numeric tensors. Here we first generate a dictionary of tensors for each label (label_tensors). We then
        # transform this to have as output of this section dictionary by image with a dictionary by labels as values
        # which can be easily queried by image in a data generator.
        # Subset labels and label types:
        label_cols = {
            label: nt for label, nt in label_cols.items() if label in label_selection
        }
        label_tensors = {}
        label_names = (
            {}
        )  # Names of individual variables in each label vector (eg. categories in onehot-encoding).
        # 1. Standardize continuous labels to z-scores:
        continuous_mean = {
            feature: tissue_meta_data[feature].mean(skipna=True)
            for feature in list(label_cols.keys())
            if label_cols[feature] == "continuous"
        }
        continuous_std = {
            feature: tissue_meta_data[feature].std(skipna=True)
            for feature in list(label_cols.keys())
            if label_cols[feature] == "continuous"
        }
        for feature in list(label_cols.keys()):
            if label_cols[feature] == "continuous":
                label_tensors[feature] = (
                    tissue_meta_data[feature].values - continuous_mean[feature]
                ) / continuous_std[feature]
                label_names[feature] = [feature]
        # 2. One-hot encode categorical columns
        # Force all entries in categorical columns to be string so that GLM-like formula processing can be performed.
        for feature in list(label_cols.keys()):
            if label_cols[feature] == "categorical":
                tissue_meta_data[feature] = tissue_meta_data[feature].astype("str")
        # One-hot encode each string label vector:
        for i, feature in enumerate(list(label_cols.keys())):
            if label_cols[feature] == "categorical":
                oh = pd.get_dummies(
                    tissue_meta_data[feature],
                    prefix=feature,
                    prefix_sep=">",
                    drop_first=False,
                )
                # Change all entries of corresponding observation to np.nan instead.
                idx_nan_col = np.array(
                    [i for i, x in enumerate(oh.columns) if x.endswith(">nan")]
                )
                if len(idx_nan_col) > 0:
                    assert len(idx_nan_col) == 1, "fatal processing error"
                    nan_rows = np.where(oh.iloc[:, idx_nan_col[0]].values == 1.0)[0]
                    oh.loc[nan_rows, :] = np.nan
                # Drop nan element column.
                oh = oh.loc[:, [x for x in oh.columns if not x.endswith(">nan")]]
                label_tensors[feature] = oh.values
                label_names[feature] = oh.columns
        # Make sure all tensors are 2D for indexing:
        for feature in list(label_tensors.keys()):
            if len(label_tensors[feature].shape) == 1:
                label_tensors[feature] = np.expand_dims(label_tensors[feature], axis=1)
        # The dictionary of tensor is nested in slices in a dictionary by image which is easier to query with a
        # generator.
        tissue_meta_data_patients = tissue_meta_data[patient_col].values.tolist()
        label_tensors = {
            img: {
                feature_name: np.array(
                    features[tissue_meta_data_patients.index(patient), :], ndmin=1
                )
                for feature_name, features in label_tensors.items()
            }
            if patient in tissue_meta_data_patients
            else None
            for img, patient in self.celldata.uns["img_to_patient_dict"].items()
        }
        # Reduce to observed patients:
        label_tensors = dict(
            [(k, v) for k, v in label_tensors.items() if v is not None]
        )

        # Save processed data to attributes.
        for k, adata in self.img_celldata.items():
            graph_covariates = {
                "label_names": label_names,
                "label_tensors": label_tensors[k],
                "label_selection": list(label_cols.keys()),
                "continuous_mean": continuous_mean,
                "continuous_std": continuous_std,
                "label_data_types": label_cols,
            }
            adata.uns["graph_covariates"] = graph_covariates

        graph_covariates = {
            "label_names": label_names,
            "label_selection": list(label_cols.keys()),
            "continuous_mean": continuous_mean,
            "continuous_std": continuous_std,
            "label_data_types": label_cols,
        }
        self.celldata.uns["graph_covariates"] = graph_covariates

        # self.ref_img_keys = {k: [] for k, v in self.nodes_by_image.items()}


class DatasetPascualReguant(Dataset):
    """DatasetPascualReguant class. Inherits all functions from Dataset."""

    cell_type_merge_dict = {
        "B cell": "B cells",
        "Endothelial cells": "Endothelial cells",
        "ILC": "ILC",
        "Monocyte/Macrohage/DC": "Monocyte/Macrohage/DC",
        "NK cell": "NK cells",
        "Plasma cell": "Plasma cells CD8",
        "T cytotoxic cell": "T cytotoxic cells",
        "T helper cell": "T helper cells",
        "other": "other",
    }

    def _register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""
        metadata = {
            "lateral_resolution": 0.325,
            "fn": [
                "TONSIL_MFI_nuclei_data_table.xlsx",
                "TONSIL_MFI_membranes_data_table.xlsx",
            ],
            "image_col": "img_keys",
            "pos_cols": ["Location_Center_X", "Location_Center_Y"],
            "cluster_col": "cell_class",
            "cluster_col_preprocessed": "cell_class_preprocessed",
            "patient_col": None,
        }
        nuclei_df = read_excel(os.path.join(self.data_path, metadata["fn"][0]))
        membranes_df = read_excel(os.path.join(self.data_path, metadata["fn"][1]))

        celldata_df = nuclei_df.join(
            membranes_df.set_index("ObjectNumber"), on="ObjectNumber"
        )

        feature_cols = [
            "Bcl6",
            "Foxp3",
            "Helios",
            "IRF4",
            "Ki67",
            "Pax5",
            "CCR6",
            "CD103",
            "CD11c",
            "CD123",
            "CD127",
            "CD138",
            "CD14",
            "CD141",
            "CD16",
            "CD161",
            "CD19",
            "CD20",
            "CD21",
            "CD23",
            "CD3",
            "CD31",
            "CD34",
            "CD38",
            "CD4",
            "CD45",
            "CD45RA",
            "CD45RO",
            "CD49a",
            "CD56",
            "CD69",
            "CD7",
            "CD8",
            "CD94",
            "CXCR3",
            "FcER1a",
            "GranzymeA",
            "HLADR",
            "ICOS",
            "IgA",
            "IgG",
            "IgM",
            "Langerin",
            "NKp44",
            "RANKL",
            "SMA",
            "TCRVa72",
            "TCRgd",
            "VCAM",
            "Vimentin",
            "cKit",
        ]
        celldata = AnnData(
            X=celldata_df[feature_cols], obs=celldata_df[["ObjectNumber", "cell_class"]]
        )

        celldata.uns["metadata"] = metadata
        celldata.obs["img_keys"] = np.repeat("tonsil_image", repeats=celldata.shape[0])
        celldata.uns["img_keys"] = ["tonsil_image"]
        # register x and y coordinates into obsm
        celldata.obsm["spatial"] = np.array(celldata_df[metadata["pos_cols"]])

        celldata.uns["img_to_patient_dict"] = {"tonsil_image": "tonsil_patient"}

        # add clean cluster column which removes regular expression from cluster_col
        celldata.obs[metadata["cluster_col_preprocessed"]] = list(
            pd.Series(
                list(celldata.obs[metadata["cluster_col"]]), dtype="category"
            ).map(self.cell_type_merge_dict)
        )
        celldata.obs[metadata["cluster_col_preprocessed"]] = celldata.obs[
            metadata["cluster_col_preprocessed"]
        ].astype("category")
        # register node type names
        node_type_names = list(
            np.unique(celldata.obs[metadata["cluster_col_preprocessed"]])
        )
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [
                node_type_names.index(x)
                for x in celldata.obs[metadata["cluster_col_preprocessed"]].values
            ]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def _register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {image key : anndata object of image}."""
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[str(k)] = self.celldata[
                self.celldata.obs[image_col] == k
            ].copy()
        self.img_celldata = img_celldata

    def _register_graph_features(self, label_selection):
        """Load graph level covariates.

        Parameters
        ----------
        label_selection
            Label selection.
        """
        # Save processed data to attributes.
        for adata in self.img_celldata.values():
            graph_covariates = {
                "label_names": {},
                "label_tensors": {},
                "label_selection": [],
                "continuous_mean": {},
                "continuous_std": {},
                "label_data_types": {},
            }
            adata.uns["graph_covariates"] = graph_covariates

        graph_covariates = {
            "label_names": {},
            "label_selection": [],
            "continuous_mean": {},
            "continuous_std": {},
            "label_data_types": {},
        }
        self.celldata.uns["graph_covariates"] = graph_covariates


class DatasetSchuerch(Dataset):
    """DatasetSchuerch class. Inherits all functions from Dataset."""

    cell_type_merge_dict = {
        "B cells": "B cells",
        "CD11b+ monocytes": "monocytes",
        "CD11b+CD68+ macrophages": "macrophages",
        "CD11c+ DCs": "dendritic cells",
        "CD163+ macrophages": "macrophages",
        "CD3+ T cells": "CD3+ T cells",
        "CD4+ T cells": "CD4+ T cells",
        "CD4+ T cells CD45RO+": "CD4+ T cells",
        "CD4+ T cells GATA3+": "CD4+ T cells",
        "CD68+ macrophages": "macrophages",
        "CD68+ macrophages GzmB+": "macrophages",
        "CD68+CD163+ macrophages": "macrophages",
        "CD8+ T cells": "CD8+ T cells",
        "NK cells": "NK cells",
        "Tregs": "Tregs",
        "adipocytes": "adipocytes",
        "dirt": "dirt",
        "granulocytes": "granulocytes",
        "immune cells": "immune cells",
        "immune cells / vasculature": "immune cells",
        "lymphatics": "lymphatics",
        "nerves": "nerves",
        "plasma cells": "plasma cells",
        "smooth muscle": "smooth muscle",
        "stroma": "stroma",
        "tumor cells": "tumor cells",
        "tumor cells / immune cells": "immune cells",
        "undefined": "undefined",
        "vasculature": "vasculature",
    }

    def _register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""
        metadata = {
            "lateral_resolution": 0.377442,
            "fn": "CRC_clusters_neighborhoods_markers_NEW.csv",
            "image_col": "File Name",
            "pos_cols": ["X:X", "Y:Y"],
            "cluster_col": "ClusterName",
            "cluster_col_preprocessed": "ClusterName_preprocessed",
            "patient_col": "patients",
        }
        celldata_df = read_csv(os.path.join(self.data_path, metadata["fn"]))

        feature_cols = [
            "CD44 - stroma:Cyc_2_ch_2",
            "FOXP3 - regulatory T cells:Cyc_2_ch_3",
            "CD8 - cytotoxic T cells:Cyc_3_ch_2",
            "p53 - tumor suppressor:Cyc_3_ch_3",
            "GATA3 - Th2 helper T cells:Cyc_3_ch_4",
            "CD45 - hematopoietic cells:Cyc_4_ch_2",
            "T-bet - Th1 cells:Cyc_4_ch_3",
            "beta-catenin - Wnt signaling:Cyc_4_ch_4",
            "HLA-DR - MHC-II:Cyc_5_ch_2",
            "PD-L1 - checkpoint:Cyc_5_ch_3",
            "Ki67 - proliferation:Cyc_5_ch_4",
            "CD45RA - naive T cells:Cyc_6_ch_2",
            "CD4 - T helper cells:Cyc_6_ch_3",
            "CD21 - DCs:Cyc_6_ch_4",
            "MUC-1 - epithelia:Cyc_7_ch_2",
            "CD30 - costimulator:Cyc_7_ch_3",
            "CD2 - T cells:Cyc_7_ch_4",
            "Vimentin - cytoplasm:Cyc_8_ch_2",
            "CD20 - B cells:Cyc_8_ch_3",
            "LAG-3 - checkpoint:Cyc_8_ch_4",
            "Na-K-ATPase - membranes:Cyc_9_ch_2",
            "CD5 - T cells:Cyc_9_ch_3",
            "IDO-1 - metabolism:Cyc_9_ch_4",
            "Cytokeratin - epithelia:Cyc_10_ch_2",
            "CD11b - macrophages:Cyc_10_ch_3",
            "CD56 - NK cells:Cyc_10_ch_4",
            "aSMA - smooth muscle:Cyc_11_ch_2",
            "BCL-2 - apoptosis:Cyc_11_ch_3",
            "CD25 - IL-2 Ra:Cyc_11_ch_4",
            "CD11c - DCs:Cyc_12_ch_3",
            "PD-1 - checkpoint:Cyc_12_ch_4",
            "Granzyme B - cytotoxicity:Cyc_13_ch_2",
            "EGFR - signaling:Cyc_13_ch_3",
            "VISTA - costimulator:Cyc_13_ch_4",
            "CD15 - granulocytes:Cyc_14_ch_2",
            "ICOS - costimulator:Cyc_14_ch_4",
            "Synaptophysin - neuroendocrine:Cyc_15_ch_3",
            "GFAP - nerves:Cyc_16_ch_2",
            "CD7 - T cells:Cyc_16_ch_3",
            "CD3 - T cells:Cyc_16_ch_4",
            "Chromogranin A - neuroendocrine:Cyc_17_ch_2",
            "CD163 - macrophages:Cyc_17_ch_3",
            "CD45RO - memory cells:Cyc_18_ch_3",
            "CD68 - macrophages:Cyc_18_ch_4",
            "CD31 - vasculature:Cyc_19_ch_3",
            "Podoplanin - lymphatics:Cyc_19_ch_4",
            "CD34 - vasculature:Cyc_20_ch_3",
            "CD38 - multifunctional:Cyc_20_ch_4",
            "CD138 - plasma cells:Cyc_21_ch_3",
            "HOECHST1:Cyc_1_ch_1",
            "CDX2 - intestinal epithelia:Cyc_2_ch_4",
            "Collagen IV - bas. memb.:Cyc_12_ch_2",
            "CD194 - CCR4 chemokine R:Cyc_14_ch_3",
            "MMP9 - matrix metalloproteinase:Cyc_15_ch_2",
            "CD71 - transferrin R:Cyc_15_ch_4",
            "CD57 - NK cells:Cyc_17_ch_4",
            "MMP12 - matrix metalloproteinase:Cyc_21_ch_4",
        ]
        feature_cols_hgnc_names = [
            "CD44",
            "FOXP3",
            "CD8A",
            "TP53",
            "GATA3",
            "PTPRC",
            "TBX21",
            "CTNNB1",
            "HLA-DR",
            "CD274",
            "MKI67",
            "PTPRC",
            "CD4",
            "CR2",
            "MUC1",
            "TNFRSF8",
            "CD2",
            "VIM",
            "MS4A1",
            "LAG3",
            "ATP1A1",
            "CD5",
            "IDO1",
            "KRT1",
            "ITGAM",
            "NCAM1",
            "ACTA1",
            "BCL2",
            "IL2RA",
            "ITGAX",
            "PDCD1",
            "GZMB",
            "EGFR",
            "VISTA",
            "FUT4",
            "ICOS",
            "SYP",
            "GFAP",
            "CD7",
            "CD247",
            "CHGA",
            "CD163",
            "PTPRC",
            "CD68",
            "PECAM1",
            "PDPN",
            "CD34",
            "CD38",
            "SDC1",
            "HOECHST1:Cyc_1_ch_1",  ##
            "CDX2",
            "COL6A1",
            "CCR4",
            "MMP9",
            "TFRC",
            "B3GAT1",
            "MMP12",
        ]
        X = DataFrame(
            np.array(celldata_df[feature_cols]), columns=feature_cols_hgnc_names
        )
        celldata = AnnData(
            X=X, obs=celldata_df[["File Name", "patients", "ClusterName"]]
        )
        celldata.var_names_make_unique()

        celldata.uns["metadata"] = metadata
        img_keys = list(np.unique(celldata_df[metadata["image_col"]]))
        celldata.uns["img_keys"] = img_keys

        celldata.obsm["spatial"] = np.array(celldata_df[metadata["pos_cols"]])

        img_to_patient_dict = {
            str(x): celldata_df[metadata["patient_col"]].values[i]
            for i, x in enumerate(celldata_df[metadata["image_col"]].values)
        }
        celldata.uns["img_to_patient_dict"] = img_to_patient_dict
        self.img_to_patient_dict = img_to_patient_dict

        # add clean cluster column which removes regular expression from cluster_col
        celldata.obs[metadata["cluster_col_preprocessed"]] = list(
            pd.Series(
                list(celldata.obs[metadata["cluster_col"]]), dtype="category"
            ).map(self.cell_type_merge_dict)
        )
        celldata.obs[metadata["cluster_col_preprocessed"]] = celldata.obs[
            metadata["cluster_col_preprocessed"]
        ].astype("category")

        # register node type names
        node_type_names = list(
            np.unique(celldata.obs[metadata["cluster_col_preprocessed"]])
        )
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [
                node_type_names.index(x)
                for x in celldata.obs[metadata["cluster_col_preprocessed"]].values
            ]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def _register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {imgage key : anndata object of image}."""
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[str(k)] = self.celldata[
                self.celldata.obs[image_col] == k
            ].copy()
        self.img_celldata = img_celldata

    def _register_graph_features(self, label_selection):
        """Load graph level covariates.

        Parameters
        ----------
        label_selection
            Label selection.
        """
        # Graph features are based on TMA spot and not patient, thus patient_col is technically wrong.
        # For aspects where patients are needed (e.g. train-val-test split) the correct patients that are
        # loaded in _register_images() are used
        patient_col = "TMA spot / region"
        disease_features = {}
        patient_features = {"Sex": "categorical", "Age": "continuous"}
        survival_features = {"DFS": "survival"}
        tumor_features = {
            # not sure where these features belong
            "Group": "categorical",
            "LA": "percentage",
            "Diffuse": "percentage",
            "Klintrup_Makinen": "categorical",
            "CLR_Graham_Appelman": "categorical",
        }
        treatment_features = {}
        col_renaming = {}

        label_cols = {}
        label_cols.update(disease_features)
        label_cols.update(patient_features)
        label_cols.update(survival_features)
        label_cols.update(tumor_features)
        label_cols.update(treatment_features)

        if label_selection is None:
            label_selection = set(label_cols.keys())
        else:
            label_selection = set(label_selection)
        label_cols_toread = list(
            label_selection.intersection(set(list(label_cols.keys())))
        )
        if "DFS" in label_selection:
            censor_col = "DFS_Censor"
            label_cols_toread = label_cols_toread + [censor_col]
        # there are two LA and Diffuse columns for the two cores that are represented by one patient row
        if "LA" in label_cols_toread:
            label_cols_toread = label_cols_toread + ["LA.1"]
        if "Diffuse" in label_cols_toread:
            label_cols_toread = label_cols_toread + ["Diffuse.1"]
        label_cols_toread_csv = [
            col_renaming[col] if col in list(col_renaming.keys()) else col
            for col in label_cols_toread
        ]

        usecols = label_cols_toread_csv + [patient_col]
        tissue_meta_data = read_csv(
            os.path.join(self.data_path, "CRC_TMAs_patient_annotations.csv"),
            # sep='\t',
            usecols=usecols,
        )[usecols]
        tissue_meta_data.columns = label_cols_toread + [patient_col]

        # preprocess the loaded csv data:
        # the rows after the first 35 are just descriptions that were included in the excel file
        # for easier work with the data, we expand the data to have two columns per patient representing the two cores
        # that have different LA and Diffuse labels
        patient_data = tissue_meta_data[:35]
        long_patient_data = pd.DataFrame(np.repeat(patient_data.values, 2, axis=0))
        long_patient_data.columns = patient_data.columns
        long_patient_data["copy"] = ["A", "B"] * 35
        if "Diffuse" in label_cols_toread:
            long_patient_data = long_patient_data.rename(
                columns={"Diffuse": "DiffuseA", "Diffuse.1": "DiffuseB"}
            )
            long_patient_data["Diffuse"] = np.zeros((70,))
            long_patient_data.loc[
                long_patient_data["copy"] == "A", "Diffuse"
            ] = long_patient_data[long_patient_data["copy"] == "A"]["DiffuseA"]
            long_patient_data.loc[
                long_patient_data["copy"] == "B", "Diffuse"
            ] = long_patient_data[long_patient_data["copy"] == "B"]["DiffuseB"]
            long_patient_data.loc[long_patient_data["Diffuse"].isnull(), "Diffuse"] = 0
            # use the proportion of diffuse cores within this spot as probability of being diffuse
            long_patient_data["Diffuse"] = (
                long_patient_data["Diffuse"].astype(float) / 2
            )
            long_patient_data = long_patient_data.drop("DiffuseA", axis=1)
            long_patient_data = long_patient_data.drop("DiffuseB", axis=1)
        if "LA" in label_cols_toread:
            long_patient_data = long_patient_data.rename(
                columns={"LA": "LAA", "LA.1": "LAB"}
            )
            long_patient_data["LA"] = np.zeros((70,))
            long_patient_data.loc[
                long_patient_data["copy"] == "A", "LA"
            ] = long_patient_data[long_patient_data["copy"] == "A"]["LAA"]
            long_patient_data.loc[
                long_patient_data["copy"] == "B", "LA"
            ] = long_patient_data[long_patient_data["copy"] == "B"]["LAB"]
            long_patient_data.loc[long_patient_data["LA"].isnull(), "LA"] = 0
            # use the proportion of LA cores within this spot as probability of being LA
            long_patient_data["LA"] = long_patient_data["LA"].astype(float) / 2
            long_patient_data = long_patient_data.drop("LAA", axis=1)
            long_patient_data = long_patient_data.drop("LAB", axis=1)
        tissue_meta_data = long_patient_data

        # BUILD LABEL VECTORS FROM LABEL COLUMNS
        # The columns contain unprocessed numeric and categorical entries that are now processed to prediction-ready
        # numeric tensors. Here we first generate a dictionary of tensors for each label (label_tensors). We then
        # transform this to have as output of this section dictionary by image with a dictionary by labels as values
        # which can be easily queried by image in a data generator.
        # Subset labels and label types:
        label_cols = {
            label: type
            for label, type in label_cols.items()
            if label in label_selection
        }
        label_tensors = {}
        label_names = (
            {}
        )  # Names of individual variables in each label vector (eg. categories in onehot-encoding).
        # 1. Standardize continuous labels to z-scores:
        continuous_mean = {
            feature: tissue_meta_data[feature].mean(skipna=True)
            for feature in list(label_cols.keys())
            if label_cols[feature] == "continuous"
        }
        continuous_std = {
            feature: tissue_meta_data[feature].std(skipna=True)
            for feature in list(label_cols.keys())
            if label_cols[feature] == "continuous"
        }
        for feature in list(label_cols.keys()):
            if label_cols[feature] == "continuous":
                label_tensors[feature] = (
                    tissue_meta_data[feature].values - continuous_mean[feature]
                ) / continuous_std[feature]
                label_names[feature] = [feature]
        for feature in list(label_cols.keys()):
            if label_cols[feature] == "percentage":
                label_tensors[feature] = tissue_meta_data[feature]
                label_names[feature] = [feature]
        # 2. One-hot encode categorical columns
        # Force all entries in categorical columns to be string so that GLM-like formula processing can be performed.
        for feature in list(label_cols.keys()):
            if label_cols[feature] == "categorical":
                tissue_meta_data[feature] = tissue_meta_data[feature].astype("str")
        # One-hot encode each string label vector:
        for i, feature in enumerate(list(label_cols.keys())):
            if label_cols[feature] == "categorical":
                oh = pd.get_dummies(
                    tissue_meta_data[feature],
                    prefix=feature,
                    prefix_sep=">",
                    drop_first=False,
                )
                # Change all entries of corresponding observation to np.nan instead.
                idx_nan_col = np.array(
                    [i for i, x in enumerate(oh.columns) if x.endswith(">nan")]
                )
                if len(idx_nan_col) > 0:
                    assert len(idx_nan_col) == 1, "fatal processing error"
                    nan_rows = np.where(oh.iloc[:, idx_nan_col[0]].values == 1.0)[0]
                    oh.loc[nan_rows, :] = np.nan
                # Drop nan element column.
                oh = oh.loc[:, [x for x in oh.columns if not x.endswith(">nan")]]
                label_tensors[feature] = oh.values
                label_names[feature] = oh.columns
        # 3. Add censoring information to survival
        survival_mean = {
            feature: tissue_meta_data[feature].mean(skipna=True)
            for feature in list(label_cols.keys())
            if label_cols[feature] == "survival"
        }
        for feature in list(label_cols.keys()):
            if label_cols[feature] == "survival":
                label_tensors[feature] = np.concatenate(
                    [
                        np.expand_dims(
                            tissue_meta_data[feature].values / survival_mean[feature],
                            axis=1,
                        ),
                        np.expand_dims(tissue_meta_data[censor_col].values, axis=1),
                    ],
                    axis=1,
                )
                label_names[feature] = [feature]
        # Make sure all tensors are 2D for indexing:
        for feature in list(label_tensors.keys()):
            if len(label_tensors[feature].shape) == 1:
                label_tensors[feature] = np.expand_dims(label_tensors[feature], axis=1)
        # The dictionary of tensor is nested in slices in a dictionary by image which is easier to query with a
        # generator.
        # tissue_meta_data_patients = tissue_meta_data[patient_col].values.tolist()
        # image keys are of the form reg0xx_A or reg0xx_B with xx going from 01 to 70
        # label tensors have entries (1+2)_A, (1+2)_B, (2+3)_A, (2+3)_B, ...
        img_to_index = {
            img: 2 * ((int(img[4:6]) - 1) // 2)
            if img[7] == "A"
            else 2 * ((int(img[4:6]) - 1) // 2) + 1
            for img in self.img_to_patient_dict.keys()
        }
        label_tensors = {
            img: {
                feature_name: np.array(features[index, :], ndmin=1)
                for feature_name, features in label_tensors.items()
            }
            for img, index in img_to_index.items()
        }

        # Save processed data to attributes.
        for k, adata in self.img_celldata.items():
            graph_covariates = {
                "label_names": label_names,
                "label_tensors": label_tensors[k],
                "label_selection": list(label_cols.keys()),
                "continuous_mean": continuous_mean,
                "continuous_std": continuous_std,
                "label_data_types": label_cols,
            }
            adata.uns["graph_covariates"] = graph_covariates

        graph_covariates = {
            "label_names": label_names,
            "label_selection": list(label_cols.keys()),
            "continuous_mean": continuous_mean,
            "continuous_std": continuous_std,
            "label_data_types": label_cols,
        }
        self.celldata.uns["graph_covariates"] = graph_covariates


class DatasetLohoff(Dataset):
    """DatasetLohoff class. Inherits all functions from Dataset."""

    cell_type_merge_dict = {
        "Allantois": "Allantois",
        "Anterior somitic tissues": "Anterior somitic tissues",
        "Blood progenitors": "Blood progenitors",
        "Cardiomyocytes": "Cardiomyocytes",
        "Cranial mesoderm": "Cranial mesoderm",
        "Definitive endoderm": "Definitive endoderm",
        "Dermomyotome": "Dermomyotome",
        "Endothelium": "Endothelium",
        "Erythroid": "Erythroid",
        "ExE endoderm": "ExE endoderm",
        "Forebrain/Midbrain/Hindbrain": "Forebrain/Midbrain/Hindbrain",
        "Gut tube": "Gut tube",
        "Haematoendothelial progenitors": "Haematoendothelial progenitors",
        "Intermediate mesoderm": "Intermediate mesoderm",
        "Lateral plate mesoderm": "Lateral plate mesoderm",
        "Low quality": "Low quality",
        "Mixed mesenchymal mesoderm": "Mixed mesenchymal mesoderm",
        "NMP": "NMP",
        "Neural crest": "Neural crest",
        "Presomitic mesoderm": "Presomitic mesoderm",
        "Sclerotome": "Sclerotome",
        "Spinal cord": "Spinal cord",
        "Splanchnic mesoderm": "Splanchnic mesoderm",
        "Surface ectoderm": "Surface ectoderm",
    }

    def _register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""
        metadata = {
            "lateral_resolution": 1.0,
            "fn": "preprocessed_lohoff.h5ad",
            "image_col": "embryo",
            "pos_cols": ["x_global", "y_global"],
            "cluster_col": "celltype_mapped_refined",
            "cluster_col_preprocessed": "celltype_mapped_refined",
            "patient_col": "embryo",
        }

        celldata = read_h5ad(os.path.join(self.data_path, metadata["fn"])).copy()
        celldata.uns["metadata"] = metadata
        celldata.uns["img_keys"] = list(np.unique(celldata.obs[metadata["image_col"]]))

        img_to_patient_dict = {
            str(x): celldata.obs[metadata["patient_col"]].values[i].split("_")[0]
            for i, x in enumerate(celldata.obs[metadata["image_col"]].values)
        }
        celldata.uns["img_to_patient_dict"] = img_to_patient_dict
        self.img_to_patient_dict = img_to_patient_dict

        # register x and y coordinates into obsm
        celldata.obsm["spatial"] = celldata.obs[metadata["pos_cols"]]

        # add clean cluster column which removes regular expression from cluster_col
        celldata.obs[metadata["cluster_col_preprocessed"]] = list(
            pd.Series(
                list(celldata.obs[metadata["cluster_col"]]), dtype="category"
            ).map(self.cell_type_merge_dict)
        )
        celldata.obs[metadata["cluster_col_preprocessed"]] = celldata.obs[
            metadata["cluster_col_preprocessed"]
        ].astype("category")
        # register node type names
        node_type_names = list(
            np.unique(celldata.obs[metadata["cluster_col_preprocessed"]])
        )
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [
                node_type_names.index(x)
                for x in celldata.obs[metadata["cluster_col_preprocessed"]].values
            ]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def _register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {imgage key : anndata object of image}."""
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[str(k)] = self.celldata[
                self.celldata.obs[image_col] == k
            ].copy()
        self.img_celldata = img_celldata

    def _register_graph_features(self, label_selection):
        """Load graph level covariates.

        Parameters
        ----------
        label_selection
            Label selection.
        """
        # Save processed data to attributes.
        for adata in self.img_celldata.values():
            graph_covariates = {
                "label_names": {},
                "label_tensors": {},
                "label_selection": [],
                "continuous_mean": {},
                "continuous_std": {},
                "label_data_types": {},
            }
            adata.uns["graph_covariates"] = graph_covariates

        graph_covariates = {
            "label_names": {},
            "label_selection": [],
            "continuous_mean": {},
            "continuous_std": {},
            "label_data_types": {},
        }
        self.celldata.uns["graph_covariates"] = graph_covariates


class DatasetLuWT(Dataset):
    """DatasetLuWT class. Inherits all functions from Dataset."""

    cell_type_merge_dict = {
        "1": "AEC",
        "2": "SEC",
        "3": "MK",
        "4": "Hepatocyte",
        "5": "Macrophage",
        "6": "Myeloid",
        "7": "Erythroid progenitor",
        "8": "Erythroid cell",
        "9": "Unknown",
    }

    def _register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""
        metadata = {
            "lateral_resolution": 0.1079,
            "fn": "FinalClusteringResults 190517 WT.csv",
            "image_col": "FOV",
            "pos_cols": ["Center_x", "Center_y"],
            "cluster_col": "CellTypeID_new",
            "cluster_col_preprocessed": "CellTypeID_new_preprocessed",
        }
        celldata_df = read_csv(os.path.join(self.data_path, metadata["fn"]))

        feature_cols = [
            "Abcb4",
            "Abcc3",
            "Adgre1",
            "Ammecr1",
            "Angpt1",
            "Angptl2",
            "Arsb",
            "Axin2",
            "B4galt6",
            "Bmp2",
            "Bmp5",
            "Bmp7",
            "Cd34",
            "Cd48",
            "Cd93",
            "Cdh11",
            "Cdh5",
            "Celsr2",
            "Clec14a",
            "Col4a1",
            "Cspg4",
            "Ctnnal1",
            "Cxadr",
            "Cxcl12",
            "Dkk2",
            "Dkk3",
            "Dll1",
            "Dll4",
            "E2f2",
            "Efnb2",
            "Egfr",
            "Egr1",
            "Eif3a",
            "Elk3",
            "Eng",
            "Ep300",
            "Epcam",
            "Ephb4",
            "Fam46c",
            "Fbxw7",
            "Fgf1",
            "Fgf2",
            "Flt3",
            "Flt4",
            "Fstl1",
            "Fzd1",
            "Fzd2",
            "Fzd3",
            "Fzd4",
            "Fzd5",
            "Fzd7",
            "Fzd8",
            "Gca",
            "Gfap",
            "Gnaz",
            "Gpd1",
            "Hc",
            "Hgf",
            "Hoxb4",
            "Icam1",
            "Igf1",
            "Il6",
            "Il7r",
            "Itga2b",
            "Itgam",
            "Jag1",
            "Jag2",
            "Kdr",
            "Kit",
            "Kitl",
            "Lef1",
            "Lepr",
            "Lox",
            "Lyve1",
            "Maml1",
            "Mecom",
            "Meis1",
            "Meis2",
            "Mertk",
            "Mki67",
            "Mmrn1",
            "Mpl",
            "Mpp1",
            "Mrc1",
            "Mrvi1",
            "Myh10",
            "Ndn",
            "Nes",
            "Nkd2",
            "Notch1",
            "Notch2",
            "Notch3",
            "Notch4",
            "Nrp1",
            "Olr1",
            "Pdgfra",
            "Pdpn",
            "Pecam1",
            "Podxl",
            "Pou2af1",
            "Prickle2",
            "Procr",
            "Proz",
            "Pzp",
            "Rassf4",
            "Rbpj",
            "Runx1",
            "Sardh",
            "Satb1",
            "Sdc3",
            "Sfrp1",
            "Sfrp2",
            "Sgms2",
            "Slamf1",
            "Slc25a37",
            "Stab2",
            "Tcf7",
            "Tcf7l1",
            "Tcf7l2",
            "Tek",
            "Tet1",
            "Tet2",
            "Tfrc",
            "Tgfb2",
            "Timp3",
            "Tmem56",
            "Tmod1",
            "Tox",
            "Vangl2",
            "Vav1",
            "Vcam1",
            "Vwf",
        ]

        celldata = AnnData(
            X=celldata_df[feature_cols],
            obs=celldata_df[
                ["CellID", "FOV", "CellTypeID_new", "Center_x", "Center_y"]
            ],
        )

        celldata.uns["metadata"] = metadata
        img_keys = list(np.unique(celldata_df[metadata["image_col"]]))
        celldata.uns["img_keys"] = img_keys

        # register x and y coordinates into obsm
        celldata.obsm["spatial"] = np.array(celldata_df[metadata["pos_cols"]])

        img_to_patient_dict = {
            str(x): "patient" for x in celldata_df[metadata["image_col"]].values
        }
        # img_to_patient_dict = {k: "p_1" for k in img_keys}
        celldata.uns["img_to_patient_dict"] = img_to_patient_dict
        self.img_to_patient_dict = img_to_patient_dict

        # add clean cluster column which removes regular expression from cluster_col
        celldata.obs[metadata["cluster_col_preprocessed"]] = list(
            pd.Series(list(celldata.obs[metadata["cluster_col"]]), dtype="str").map(
                self.cell_type_merge_dict
            )
        )
        celldata.obs[metadata["cluster_col_preprocessed"]] = celldata.obs[
            metadata["cluster_col_preprocessed"]
        ].astype("str")
        # celldata = celldata[celldata.obs[metadata["cluster_col_preprocessed"]] != 'Unknown']

        # register node type names
        node_type_names = list(
            np.unique(celldata.obs[metadata["cluster_col_preprocessed"]])
        )
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [
                node_type_names.index(x)
                for x in celldata.obs[metadata["cluster_col_preprocessed"]].values
            ]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def _register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {imgage key : anndata object of image}."""
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[str(k)] = self.celldata[
                self.celldata.obs[image_col] == k
            ].copy()
        self.img_celldata = img_celldata

    def _register_graph_features(self, label_selection):
        """Load graph level covariates.

        Parameters
        ----------
        label_selection
            Label selection.
        """
        # Save processed data to attributes.
        for adata in self.img_celldata.values():
            graph_covariates = {
                "label_names": {},
                "label_tensors": {},
                "label_selection": [],
                "continuous_mean": {},
                "continuous_std": {},
                "label_data_types": {},
            }
            adata.uns["graph_covariates"] = graph_covariates

        graph_covariates = {
            "label_names": {},
            "label_selection": [],
            "continuous_mean": {},
            "continuous_std": {},
            "label_data_types": {},
        }
        self.celldata.uns["graph_covariates"] = graph_covariates


class DatasetLuTET2(Dataset):
    """DatasetLuTET2 class. Inherits all functions from Dataset."""

    cell_type_merge_dict = {
        "1": "AEC",
        "2": "SEC",
        "3": "MK",
        "4": "Hepatocyte",
        "5": "Macrophage",
        "6": "Myeloid",
        "7": "Erythroid progenitor",
        "8": "Erythroid cell",
        "9": "Unknown",
    }

    def _register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""
        metadata = {
            "lateral_resolution": 0.1079,
            "fn": "FinalClusteringResults 190727 TET2.csv",
            "image_col": "FOV",
            "pos_cols": ["Center_x", "Center_y"],
            "cluster_col": "CellTypeID_new",
            "cluster_col_preprocessed": "CellTypeID_new_preprocessed",
        }
        celldata_df = read_csv(os.path.join(self.data_path, metadata["fn"]))

        feature_cols = [
            "Abcb4",
            "Abcc3",
            "Adgre1",
            "Ammecr1",
            "Angpt1",
            "Angptl2",
            "Arsb",
            "Axin2",
            "B4galt6",
            "Bmp2",
            "Bmp5",
            "Bmp7",
            "Cd34",
            "Cd48",
            "Cd93",
            "Cdh11",
            "Cdh5",
            "Celsr2",
            "Clec14a",
            "Col4a1",
            "Cspg4",
            "Ctnnal1",
            "Cxadr",
            "Cxcl12",
            "Dkk2",
            "Dkk3",
            "Dll1",
            "Dll4",
            "E2f2",
            "Efnb2",
            "Egfr",
            "Egr1",
            "Eif3a",
            "Elk3",
            "Eng",
            "Ep300",
            "Epcam",
            "Ephb4",
            "Fam46c",
            "Fbxw7",
            "Fgf1",
            "Fgf2",
            "Flt3",
            "Flt4",
            "Fstl1",
            "Fzd1",
            "Fzd2",
            "Fzd3",
            "Fzd4",
            "Fzd5",
            "Fzd7",
            "Fzd8",
            "Gca",
            "Gfap",
            "Gnaz",
            "Gpd1",
            "Hc",
            "Hgf",
            "Hoxb4",
            "Icam1",
            "Igf1",
            "Il6",
            "Il7r",
            "Itga2b",
            "Itgam",
            "Jag1",
            "Jag2",
            "Kdr",
            "Kit",
            "Kitl",
            "Lef1",
            "Lepr",
            "Lox",
            "Lyve1",
            "Maml1",
            "Mecom",
            "Meis1",
            "Meis2",
            "Mertk",
            "Mki67",
            "Mmrn1",
            "Mpl",
            "Mpp1",
            "Mrc1",
            "Mrvi1",
            "Myh10",
            "Ndn",
            "Nes",
            "Nkd2",
            "Notch1",
            "Notch2",
            "Notch3",
            "Notch4",
            "Nrp1",
            "Olr1",
            "Pdgfra",
            "Pdpn",
            "Pecam1",
            "Podxl",
            "Pou2af1",
            "Prickle2",
            "Procr",
            "Proz",
            "Pzp",
            "Rassf4",
            "Rbpj",
            "Runx1",
            "Sardh",
            "Satb1",
            "Sdc3",
            "Sfrp1",
            "Sfrp2",
            "Sgms2",
            "Slamf1",
            "Slc25a37",
            "Stab2",
            "Tcf7",
            "Tcf7l1",
            "Tcf7l2",
            "Tek",
            "Tet1",
            "Tet2",
            "Tfrc",
            "Tgfb2",
            "Timp3",
            "Tmem56",
            "Tmod1",
            "Tox",
            "Vangl2",
            "Vav1",
            "Vcam1",
            "Vwf",
        ]

        celldata = AnnData(
            X=celldata_df[feature_cols],
            obs=celldata_df[
                ["CellID", "FOV", "CellTypeID_new", "Center_x", "Center_y"]
            ],
        )

        celldata.uns["metadata"] = metadata
        img_keys = list(np.unique(celldata_df[metadata["image_col"]]))
        celldata.uns["img_keys"] = img_keys

        # register x and y coordinates into obsm
        celldata.obsm["spatial"] = np.array(celldata_df[metadata["pos_cols"]])

        img_to_patient_dict = {
            str(x): "patient" for x in celldata_df[metadata["image_col"]].values
        }
        # img_to_patient_dict = {k: "p_1" for k in img_keys}
        celldata.uns["img_to_patient_dict"] = img_to_patient_dict
        self.img_to_patient_dict = img_to_patient_dict

        # add clean cluster column which removes regular expression from cluster_col
        celldata.obs[metadata["cluster_col_preprocessed"]] = list(
            pd.Series(list(celldata.obs[metadata["cluster_col"]]), dtype="str").map(
                self.cell_type_merge_dict
            )
        )
        celldata.obs[metadata["cluster_col_preprocessed"]] = celldata.obs[
            metadata["cluster_col_preprocessed"]
        ].astype("str")
        # register node type names
        node_type_names = list(
            np.unique(celldata.obs[metadata["cluster_col_preprocessed"]])
        )
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [
                node_type_names.index(x)
                for x in celldata.obs[metadata["cluster_col_preprocessed"]].values
            ]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def _register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {imgage key : anndata object of image}."""
        image_col = self.celldata.uns["metadata"]["image_col"]
        img_celldata = {}
        for k in self.celldata.uns["img_keys"]:
            img_celldata[str(k)] = self.celldata[
                self.celldata.obs[image_col] == k
            ].copy()
        self.img_celldata = img_celldata

    def _register_graph_features(self, label_selection):
        """Load graph level covariates.

        Parameters
        ----------
        label_selection
            Label selection.
        """
        # Save processed data to attributes.
        for adata in self.img_celldata.values():
            graph_covariates = {
                "label_names": {},
                "label_tensors": {},
                "label_selection": [],
                "continuous_mean": {},
                "continuous_std": {},
                "label_data_types": {},
            }
            adata.uns["graph_covariates"] = graph_covariates

        graph_covariates = {
            "label_names": {},
            "label_selection": [],
            "continuous_mean": {},
            "continuous_std": {},
            "label_data_types": {},
        }
        self.celldata.uns["graph_covariates"] = graph_covariates


class DatasetSalasIss(Dataset):
    """Dataset10xVisiumMouseBrain class. Inherits all functions from Dataset."""

    cell_type_merge_dict = {
        "x_0_": "Mesenchymal",
        "x_1_": "Mesenchymal",
        "x_2_": "Mesenchymal",
        "x_3_": "Mesenchymal",
        "x_4_": "Prol. Mesench.",
        "x_5_": "Mesenchymal",
        "x_6_": "Mesenchymal",
        "x_7_": "Mesenchymal",
        "x_8_": "ASM",
        "x_9_": "immature ASM",
        "x_10_": "Vasc. Endothelial",
        "x_11_": "Prol. Mesench.",
        "x_12_": "Epith. Proximal",
        "x_13_": "Epith. Distal",
        "x_14_": "Mesenchymal",
        "x_15_": "Prol. Mesench.",
        "x_16_": "Pericytes",
        "x_17_": "Immune myeloid",
        "x_18_": "Prol. Mesench.",
        "x_19_": "Chondroblasts",
        "x_20_": "Mesothelial",
        "x_21_": "Erythroblast-RBC",
        "x_22_": "Immune Lymphoid",
        "x_23_": "Neuronal",
        "x_24_": "Epithelial NE",
        "x_25_": "Mesenchymal",
        "x_26_": "Lymph. Endoth.",
        "x_27_": "Epi. Ciliated",
        "x_28_": "Megacaryocyte",
        "x_29_": "Mesenchymal",
    }

    def _register_celldata(self, n_top_genes: Optional[int] = None):
        """Load AnnData object of complete dataset."""
        metadata = {
            "lateral_resolution": 1.0,
            "fn": "Cell_type/PCW13/S1T1_pciseq_mALL_pciseq_results_V02.csv",
            "cluster_col": "name",
            "cluster_col_preprocessed": "name_preprocessed",
            "pos_cols": ["X", "Y"],
        }

        celldata_df = read_csv(os.path.join(self.data_path, metadata["fn"]))

        feature_cols = [
            "BCL2",
            "CCL21",
            "CDON",
            "CPM",
            "CTGF",
            "CTNND2",
            "CXXC4",
            "DLL1",
            "DLL3",
            "DLL4",
            "DNAH12",
            "DTX1",
            "ETS1",
            "ETS2",
            "ETV1",
            "ETV3",
            "ETV5",
            "ETV6",
            "FGF10",
            "FGF18",
            "FGF2",
            "FGF20",
            "FGF7",
            "FGF9",
            "FGFR1",
            "FGFR2",
            "FGFR3",
            "FGFR4",
            "FLT1",
            "FLT4",
            "FZD1",
            "FZD2",
            "FZD3",
            "FZD6",
            "FZD7",
            "GLI2",
            "GLI3",
            "GRP",
            "HES1",
            "HEY1",
            "HEYL",
            "HGF",
            "IGFBP7",
            "JAG1",
            "JAG2",
            "KDR",
            "LEF1",
            "LRP2",
            "LRP5",
            "LYZ",
            "MET",
            "MFNG",
            "NKD1",
            "NOTCH1",
            "NOTCH2",
            "NOTCH3",
            "NOTCH4",
            "NOTUM",
            "PDGFA",
            "PDGFB",
            "PDGFC",
            "PDGFD",
            "PDGFRA",
            "PDGFRB",
            "PHOX2B",
            "PTCH1",
            "RSPH1",
            "RSPO2",
            "RSPO3",
            "SEC11C",
            "SFRP1",
            "SFRP2",
            "SHH",
            "SMO",
            "SOX17",
            "SPOPL",
            "SPRY1",
            "SPRY2",
            "TCF7L1",
            "TPPP3",
            "VEGFB",
            "VEGFC",
            "VEGFD",
            "WIF1",
            "WNT11",
            "WNT2",
            "WNT2B",
            "WNT5A",
            "WNT7B",
        ]
        X = DataFrame(np.array(celldata_df[feature_cols]), columns=feature_cols)
        celldata = AnnData(X=X, obs=celldata_df[["name"]])
        celldata.var_names_make_unique()

        celldata.uns["metadata"] = metadata

        celldata.obsm["spatial"] = np.array(celldata_df[metadata["pos_cols"]])

        img_to_patient_dict = {"image": "patient"}
        celldata.uns["img_to_patient_dict"] = img_to_patient_dict
        self.img_to_patient_dict = img_to_patient_dict

        # add clean cluster column which removes regular expression from cluster_col
        celldata.obs[metadata["cluster_col_preprocessed"]] = list(
            pd.Series(
                list(celldata.obs[metadata["cluster_col"]]), dtype="category"
            ).map(self.cell_type_merge_dict)
        )
        celldata.obs[metadata["cluster_col_preprocessed"]] = celldata.obs[
            metadata["cluster_col_preprocessed"]
        ].astype("category")

        # register node type names
        node_type_names = list(
            np.unique(celldata.obs[metadata["cluster_col_preprocessed"]])
        )
        celldata.uns["node_type_names"] = {x: x for x in node_type_names}
        node_types = np.zeros((celldata.shape[0], len(node_type_names)))
        node_type_idx = np.array(
            [
                node_type_names.index(x)
                for x in celldata.obs[metadata["cluster_col_preprocessed"]].values
            ]  # index in encoding vector
        )
        node_types[np.arange(0, node_type_idx.shape[0]), node_type_idx] = 1
        celldata.obsm["node_types"] = node_types

        self.celldata = celldata

    def _register_img_celldata(self):
        """Load dictionary of of image-wise celldata objects with {imgage key : anndata object of image}."""
        img_celldata = {}
        self.img_celldata = {"image": self.celldata}

    def _register_graph_features(self, label_selection):
        """Load graph level covariates.

        Parameters
        ----------
        label_selection
            Label selection.
        """
        # Save processed data to attributes.
        for adata in self.img_celldata.values():
            graph_covariates = {
                "label_names": {},
                "label_tensors": {},
                "label_selection": [],
                "continuous_mean": {},
                "continuous_std": {},
                "label_data_types": {},
            }
            adata.uns["graph_covariates"] = graph_covariates

        graph_covariates = {
            "label_names": {},
            "label_selection": [],
            "continuous_mean": {},
            "continuous_std": {},
            "label_data_types": {},
        }
        self.celldata.uns["graph_covariates"] = graph_covariates
