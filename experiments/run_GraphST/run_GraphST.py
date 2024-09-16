import scanpy as sc
import pandas as pd
import math
from scipy.sparse import csr_matrix
import numpy as np
from sklearn import metrics
import torch
import argparse
from datetime import datetime
import gc

# Plotting

import seaborn as sns

# System
from pathlib import Path
import os
from GraphST import GraphST
from GraphST.utils import clustering
import itertools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyreadr

# Ensure you are always in the parent dir
os.chdir('/home/kyan/git/cv-scdl3991/')
data_path = Path('data/')

import warnings
warnings.simplefilter("ignore")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# setting seed
torch.manual_seed(17)


# Loading all adatas
# Function to load all valid DLPC datasets from the DPLC directory
def load_dlpc_datasets(dlpc_dir):
    dlpc_dir = Path(dlpc_dir)
    datasets = []

    # Iterate through all directories in the DPLC folder
    for folder in dlpc_dir.iterdir():
        if folder.is_dir():  # Check if it's a directory
            patient_id = folder.name
            count_file = folder / (patient_id + "_filtered_feature_bc_matrix.h5")
            if count_file.exists():  # Check if the data file exists
                print(f"Loading data from {folder.name}...")
                adata = sc.read_visium(folder, count_file=f"{patient_id}_filtered_feature_bc_matrix.h5")
                adata.uns['name'] = folder.name
                datasets.append(adata)
            else:
                print(f"Skipping folder {folder.name}: no valid data file found.")
    
    return datasets


def load_MH_datasets(MH_dir):
    datasets = []
    # Files must follow the naming scheme MH_{sample}.h5ad
    for file in MH_dir.glob("MH_*.h5ad"):
        print(f"Loading data from {file.stem}...")
        adata = sc.read_h5ad(file)
        adata.uns['name'] = file.stem
        datasets.append(adata)
    return datasets


# Load the specific datasets
def load_datasets():

    datasets = []

    data_path_MSC = Path('data/MSC/MSC_gene_expression_FINAL.h5ad')
    adata = sc.read_h5ad(data_path_MSC)
    adata.uns['name'] = 'MSC'

    data_path_HBCA1 = Path('data/HBCA1/')
    adata1 = sc.read_visium(data_path_HBCA1, count_file="V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5")
    adata1.uns['name'] = 'HBCA1'
    
    dlpc_adatas = load_dlpc_datasets(data_path/"DLPC")
    MH_adatas = load_MH_datasets(data_path/"MH")

    datasets.extend([adata, adata1])
    datasets.extend(dlpc_adatas)
    datasets.extend(MH_adatas)

    return datasets


# Loading all ground truths:
def load_dlpc_ground_truths(dlpc_dir):
    dlpc_dir = Path(dlpc_dir)
    gt_paths = {}
    for folder in dlpc_dir.iterdir():
        if folder.is_dir():
            patient_id = folder.name
            gt_file = folder/ "gt/tissue_positions_list_GTs.txt"
            
            if gt_file.exists():  # Check if the ground truth file exists
                gt_paths[patient_id] = str(gt_file)
    return gt_paths

def process_DLPC_gt(path):
    gt = pd.read_csv(path, header=None)
    gt['gt'] = gt.iloc[:, -1]
    gt = gt.set_index(0)
    gt = gt[['gt']]
    return gt

def process_HBCA1_gt(path):
    df_meta = pd.read_csv(path, sep='\t')
    df_meta.set_index('ID', inplace=True)
    df_meta['gt'] = pd.Categorical(df_meta.iloc[:, -1]).codes
    return df_meta[['gt']]

def process_MSC_gt(adata):
    return pd.DataFrame(adata.obs['ClusterID']).rename(columns={'ClusterID': 'gt'})

def process_MH_gt(adata):
    adata.obs['gt'] = pd.Categorical(adata.obs['Cell_class']).codes
    return adata.obs[['gt']]

def load_all_ground_truths(gt_paths, adata_list):
    ground_truths = {}

    for adata in adata_list:
        try:
            adata_name = adata.uns['name']
            
            if adata_name == 'HBCA1':
                ground_truths[adata_name] = process_HBCA1_gt(gt_paths[adata_name])
            elif adata_name.startswith('151'):
                ground_truths[adata_name] = process_DLPC_gt(gt_paths[adata_name])
            elif adata_name == 'MSC':
                ground_truths[adata_name] = process_MSC_gt(adata)
            elif adata_name.startswith('MH'):
                ground_truths[adata_name] = process_MH_gt(adata)
        except:
            print(f"An error has occurred with file {adata_name}, skipping to next file.")

    return ground_truths

#### Clustering

def run_clustering(adata, gt, **kwargs):
    """
    Runs clustering on the AnnData object, computes metrics, and returns results.

      Parameters
    ----------
    adata : AnnData
        The single-cell data stored in an AnnData object, which will be used for clustering.
    gt : Dataframe containing ground truths
    **kwargs : dict, optional
        Optional parameters for clustering. The following keys can be passed:
        - 'n_clusters' (int): Number of clusters for the clustering algorithm.
        - 'radius' (float, default=50): The radius parameter for spatial clustering, if applicable.
        - 'tool' (str, default='mclust'): The clustering method to use. Options are 'mclust', 'leiden', or 'louvain'.
        - 'refinement' (bool, default=False): Whether to apply a refinement step during clustering.

    Returns
    -------
    adata : AnnData
        The modified AnnData object with clustering results and ground truth labels merged in the `obs` DataFrame.
    clusters : pd.Series
        The series of predicted cluster labels from the clustering algorithm.
    gt : pd.Series
        The series of ground truth labels for the corresponding cells in `adata`.
    """

    adata = adata.copy()
    model = GraphST.GraphST(adata, device=device)
    adata = model.train()

    n_clusters = kwargs.get('n_clusters')
    radius = kwargs.get('radius', 50)
    tool = kwargs.get('tool', 'mclust')  # default to 'mclust' if not provided
    refinement = kwargs.get('refinement', False)

    if tool == 'mclust':
       clustering(adata, n_clusters, radius=radius, method=tool, refinement=refinement) # For DLPFC dataset, we use optional refinement step.
    elif tool in ['leiden', 'louvain']:
       clustering(adata, n_clusters, radius=radius, method=tool, start=0.1, end=2.0, increment=0.01, refinement=False)
    
    merged_df = pd.merge(adata.obs, gt, left_index=True, right_index=True, how='outer')
    adata.obs = merged_df
    
    # filter for missing values
    adata = adata[~pd.isnull(adata.obs['gt'])]
    clusters = adata.obs['domain']
    gt = adata.obs['gt']

    return adata, clusters, gt

def run_all_clustering(adata, gt, results_path, adata_path, **kwargs):
    """
    Runs the clustering process and saves results to disk.
    
    Args:
        adata (AnnData): The AnnData object containing the dataset.
        gt (array-like): The ground truth labels for clustering.
        **kwargs: Additional keyword arguments for clustering, such as 'n_clusters'.

    Returns:
        None
    """
    # Run clustering and compute metrics
    adata, clusters, gt = run_clustering(adata, gt, **kwargs)
    dataset_row, columns = compute_metrics(adata.uns['name'], adata, clusters, gt)

    # Prepare the new dataframe for saving
    n_clusters = kwargs.get('n_clusters')
    new_df = prepare_new_dataframe(dataset_row, columns, n_clusters)

    # Load existing results and handle duplication of columns
    saved_df = load_existing_results(results_path)
    check_duplicate_columns(new_df)

    # Combine and save updated results
    result = merge_dataframes(saved_df, new_df)
    save_results(result, results_path)

    # Save the updated AnnData object with clustering results
    save_clustering_results(adata, adata_path)

    # Free up memory by deleting.
    del adata
    gc.collect()


def run_dataset_clustering(adata, gt, results_path, adata_path, **kwargs):
    """
    Runs clustering for a single dataset and saves the results.

    Args:
        adata (AnnData): The AnnData object containing the dataset.
        gt (pd.DataFrame): The ground truth labels for clustering.
        results_path (str or Path): Path to save the clustering results.
        adata_path (str or Path): Path to save the AnnData object with clustering results.

    Returns:
        None
    """
    n_clusters = len(gt['gt'].unique())
    for clusters in range(n_clusters-3, n_clusters+3, 2):
        try:
            run_all_clustering(adata, gt, results_path, adata_path, **kwargs)
        except Exception as e:
            print(f"Error clustering dataset {adata.uns['name']} with {clusters} clusters: {e}")

def compute_metrics(name, adata, clusters, gt):
    adata = adata[~pd.isnull(adata.obs['gt'])]
    X = adata.X
    # 6 metrics in total
    ARI = metrics.adjusted_rand_score(clusters, gt)
    AMI = metrics.adjusted_mutual_info_score(clusters, gt)
    HOM = metrics.homogeneity_score(clusters, gt)

    # Unsupervised metrics
    SIL = metrics.silhouette_score(X, clusters)
    CH = metrics.calinski_harabasz_score(X.toarray(), clusters)
    DBI = metrics.davies_bouldin_score(X.toarray(), clusters)
    
    return {name: [ARI, AMI, HOM, SIL, CH, DBI]}, ["ARI", "AMI", "HOM", "SIL", "CH", "DBI"]

def prepare_new_dataframe(dataset_row, columns, n_clusters):
    """
    Prepares the new dataframe to append clustering results.
    
    Args:
        dataset_row (dict): Row of clustering results.
        columns (list): List of columns for the dataframe.
        n_clusters (int): Number of clusters used for clustering.

    Returns:
        pd.DataFrame: The new dataframe with clustering metadata added.
    """
    new_df = pd.DataFrame.from_dict(dataset_row, columns=columns, orient='index')
    new_df["n_clusters"] = n_clusters
    new_df["method"] = "GraphST"
    new_df["method_type"] = "JOINT LOW DIMENSIONAL SPACE DETECTION"
    print(f"Adding entry {new_df.index} to dataset")
    return new_df

def load_existing_results(results_path):
    """
    Loads the saved dataframe from the disk.

    Args:
        results_path (str or Path): Path to the saved results.

    Returns:
        pd.DataFrame: The saved dataframe.
    """
    try:
        return pd.read_csv(results_path, index_col=0)
    except FileNotFoundError:
        print(f"Results file not found at {results_path}. Returning empty dataframe.")
        return pd.DataFrame()

def check_duplicate_columns(new_df):
    """
    Checks for and prints warnings if duplicate columns exist in the new dataframe.

    Args:
        new_df (pd.DataFrame): The new dataframe to check.

    Returns:
        None
    """
    duplicate_columns = new_df.columns[new_df.columns.duplicated()]
    if duplicate_columns.any():
        print(f"WARNING: Duplicate columns in {new_df.index}: {duplicate_columns}")

def merge_dataframes(saved_df, new_df):
    """
    Merges the existing dataframe with the new dataframe, ensuring common columns are retained.

    Args:
        saved_df (pd.DataFrame): The previously saved dataframe.
        new_df (pd.DataFrame): The new dataframe to append.

    Returns:
        pd.DataFrame: The combined dataframe.
    """
    common_columns = saved_df.columns.intersection(new_df.columns)
    print(f"Common Columns: {common_columns} \n")
    saved_df_common = saved_df[common_columns]
    new_df_common = new_df[common_columns]
    return pd.concat([saved_df_common, new_df_common], join="inner")

def save_results(result, results_path):
    """
    Saves the combined results to disk.

    Args:
        result (pd.DataFrame): The final dataframe to save.
        results_path (str or Path): Path to save the results.

    Returns:
        None
    """
    save_path = Path(results_path)
    result.to_csv(save_path, index=True)
    print(f"Results saved to {save_path}")

def save_clustering_results(adata, adata_path):
    """
    Saves the AnnData object with clustering results.

    Args:
        adata (AnnData): The AnnData object with clustering results.
        adata_path (str or Path): Path to save the AnnData object.

    Returns:
        None
    """
    adata.X = csr_matrix(adata.X)  # Convert dense matrix to sparse format
    filename = f"{adata.uns['name']}_clustering_results.h5ad"
    adata.write_h5ad(Path(adata_path) / filename)
    print(f"AnnData saved to {adata_path}/{filename}")

def setup_save_path():
    """Ensures the save directory exists and returns the save path. Also creates a path to store all the modified anndatas"""
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%H-%M")

    path = f"outputs/clustering/GraphST-{dt_string}/clustering.csv"
    path1 = f"outputs/clustering/GraphST-{dt_string}/adatas"

    save_dir = Path(path).parent

    # Creating both the save directory and the anndata directory
    save_dir.mkdir(parents=True, exist_ok=True)
    Path(path1).mkdir(parents=True, exist_ok=True)

    return path, path1


def main():
    # Getting paths to ground truths for all
    gt_paths = load_dlpc_ground_truths(data_path/"DLPC")
    gt_paths["HBCA1"] = "data/HBCA1/gt/gold_metadata.tsv"
    datasets = load_datasets()
    ground_truths = load_all_ground_truths(gt_paths, datasets)
    
    parser = argparse.ArgumentParser(description="Run GraphST pipeline for all AnnData objects.")
    parser.add_argument('--all', action='store_true', help='Run GraphST pipeline on all stored datasets')
    parser.add_argument('--path', action='store', help='Save path for output file')

    args = parser.parse_args()

    if args.path:
        save_path = args.path
    else:
        results_path, adata_path = setup_save_path()

    # Create an empty DataFrame with the specified columns
    columns = ['ARI', 'AMI', 'HOM', 'SIL', 'CH', 'DBI', 'n_clusters', 'method', 'method_type']
    empty_df = pd.DataFrame(columns=columns)
    empty_df.to_csv(results_path, index=False)
    
    # # Ensure the correct start method for multiprocessing
    # multiprocessing.set_start_method("spawn", force=True)

    # if args.all:
    #     try:
    #         max_workers = 4
    #         with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #             # Prepare a list of tasks
    #             futures = []
    #             for adata in datasets:
    #                 adata_name = adata.uns['name']
    #                 gt = ground_truths[adata_name]
    #                 futures.append(
    #                     executor.submit(run_dataset_clustering, adata, gt, results_path, adata_path)
    #                 )

    #             # Monitor the progress of each clustering task
    #             for future in as_completed(futures):
    #                 try:
    #                     future.result()  # This will raise any exception encountered during execution
    #                 except Exception as e:
    #                     print(f"Error during clustering: {e}")

    #     except Exception as e:
    #         print(f"Error running clustering: {e}")

    
    if args.all:
        first = True
        for adata in datasets:
            try:
                if first:
                    n_clusters = len(ground_truths[adata.uns["name"]]["gt"].unique())
                    gt = ground_truths[adata.uns["name"]]
                    for clusters in range(n_clusters-1, n_clusters+2, 1):
                        if first:
                            adata_res, clusters, gt = run_clustering(adata, gt, n_clusters=clusters, radius=50, refinement= True)
                            dataset_row, columns = compute_metrics(adata.uns['name'], adata_res, clusters, gt)
                            new_df = prepare_new_dataframe(dataset_row, columns, n_clusters= clusters)
                            save_results(new_df, results_path)
                            # Save the updated AnnData object with clustering results
                            save_clustering_results(adata, adata_path)
                            first=False
                        else:
                            run_all_clustering(adata, gt, results_path, adata_path, n_clusters=clusters, refinement="True")
                else:
                    n_clusters = len(ground_truths[adata.uns["name"]]["gt"].unique())
                    for clusters in range(n_clusters-1, n_clusters+2, 1):
                        gt = ground_truths[adata.uns["name"]]
                        run_all_clustering(adata, gt, results_path, adata_path, n_clusters=clusters, refinement="True")
            except:
                print(f"Error clustering dataset {adata.uns["name"]}")
                pass 
            finally:
            # Free up memory by deleting the AnnData object
                del adata
            # Force garbage collection
                gc.collect()

if __name__ == "__main__":
    main()