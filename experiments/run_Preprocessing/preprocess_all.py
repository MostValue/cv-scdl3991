import scanpy as sc
import pandas as pd
import math
import numpy as np
import argparse
# System
from pathlib import Path
import os


import warnings
warnings.simplefilter("ignore")


# Ensure you are always in the parent dir
os.chdir('/home/kyan/git/cv-scdl3991/')
data_path = Path('data/')


def setup_qc_metrics(adata):
    """
    Sets up and performs doublet detection for an AnnData object.

    1. Identifies the top 2000 highly variable genes using the 'seurat_v3' method 
    2. Uses the scVI model to detect and remove doublets
    3. Trains a SOLO model to further refine the detection of doublets.
   
    Parameters
    ----------
    adata : AnnData
        An AnnData object containing single-cell gene expression data.

    Returns
    -------
    AnnData
        The input AnnData object with additional QC metrics, including doublet predictions, stored in the `obs` DataFrame.
    """
    # setup

    adata.var_names_make_unique()

    sc.pp.highly_variable_genes(adata, n_top_genes = 2000, subset=True, flavor = 'seurat_v3', inplace=True)

    # # using scvi to remove doublets
    # scvi.model.SCVI.setup_anndata(adata)
    # vae = scvi.model.SCVI(adata)
    # vae.train()

    # # training a solo model
    # solo = scvi.external.SOLO.from_scvi_model(vae)
    # solo.train()

    # temp_df = solo.predict()
    # temp_df['predict'] = solo.predict(soft = False)
    # temp_df['difference'] = temp_df.doublet - temp_df.singlet 
    # doublets = temp_df[(temp_df['predict'] == 'doublet') & (temp_df['difference'] > 1)] # filtering out only those that have a predicted difference of more than 1

    # # adding doublet prediction to adata object
    # adata.obs['doublet'] = adata.obs.index.isin(doublets)

    return adata


def compute_qc_metrics(adata):
    """
    Compute quality control (QC) metrics for an AnnData object.

    This function calculates various QC metrics for the input AnnData object, such as the percentage of counts 
    assigned to specific gene categories (mitochondrial, ribosomal, hemoglobin genes), and the percentage of counts 
    assigned to the top x% of genes per cell. Additionally, it computes the percentage of highly variable genes.

    Parameters
    ----------
    adata : AnnData
        An AnnData object containing single-cell gene expression data. The input object should have a `var` DataFrame 
        with gene names.

    Returns
    -------
    AnnData
        The input AnnData object with additional QC metrics stored in the `obs` and `var` DataFrames.
    """
    
    # default is no filtering 
    # sc.pp.filter_cells(adata, min_genes = 200) 
    
    adata.var["mt"] = adata.var_names.str.startswith(("MT-", "mt-", "Mt-"))
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var["hb"] = adata.var_names.str.contains(("^HB[^(P)]"))


    # To calculate the % of counts assigned to the top x genes per cell. Some data have very small cell counts so % were used instead
    n_genes = len(adata.var)
    percents = [0.01, 0.05, 0.1, 0.2] # 1% , 5%, 10%, 20% of genes)
    ceiling_values = [math.ceil(n_genes * p) for p in percents]
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, percent_top=ceiling_values, log1p=True)

    # Doing some renaming
    percent_names = [f"pct_counts_in_top_{n}_genes" for n in ceiling_values]
    new_names = [f"pct_counts_in_top_{round(n*100)}_pct_genes" for n in percents]
    adata.obs.rename(columns = dict(zip(percent_names, new_names)), inplace = True)

    rename_totals = {'total_counts': 'total_counts_genes', 'log1p_total_counts': 'log1p_total_counts_genes'}
    adata.var.rename(columns = rename_totals, inplace = True)
    # remove = ['total_counts_mt', 'log1p_total_counts_mt', 'total_counts_ribo', 
    #       'log1p_total_counts_ribo', 'total_counts_hb', 'log1p_total_counts_hb']
    
    # adata.obs = adata.obs[[x for x in adata.obs.columns if x not in remove]]
    
    #### Other Metrics #####
    # adata.var['pct_highly_var_genes'] = adata.var["highly_variable"].sum() / len(adata.var). Realised this is not a useful metric
    
    return adata

def extract_metrics(adata, agg=np.median, exclude = None):
    """
    Extracts summary statistics from an AnnData object.
    
    Parameters:
    - adata: AnnData object, after highly variable genes, and doublets have been identified.
    - agg: function to apply to each variable (default: np.mean)
    
    Returns:
    - dict, column_names: dictionary of {row: [metrics]}, list of all column_names
    """
    try:
        index = adata.uns['name']
    except:
        print("No name set for AnnData object")

    ##### Computing metrics for all cells. #####

    # selecting only numeric quantities
    numeric_obs = adata.obs.select_dtypes(include=['number']).columns.tolist()
    numeric_obs = [item for item in numeric_obs if not item.startswith("_")]

    # applying our agg function
    obs_metrics = adata.obs[numeric_obs].apply(agg).to_list()

    ##### Computing metrics for all genes.#####
    
    # selecting only numeric quantities
    numeric_vars = adata.var.select_dtypes(include=['number']).columns.tolist()

    # EXCLUDE
    exclude = ['highly_variable_rank']
    
    numeric_vars = [item for item in numeric_vars if ((not item.startswith("_")) and (item not in exclude))]

    # applying our agg function
    var_metrics = adata.var[numeric_vars].apply(agg).to_list()

    ##### Custom defined metrics #####

    ## TODO

    ##### PCA Extraction #####

    obs_metrics.extend(var_metrics)
    numeric_obs.extend(numeric_vars)
    
    return {index: obs_metrics}, numeric_obs


def run_all_qc_partially(adata, save_path):
    save_path = "outputs/characteristics/characteristics.csv"
    
    adata = setup_qc_metrics(adata)
    adata = compute_qc_metrics(adata)
    dataset_row, columns = extract_metrics(adata)
    new_df = pd.DataFrame.from_dict(dataset_row, columns = columns, 
                                    orient='index') 
    return new_df


def run_all_qc(adata, save_path):
    
    adata = setup_qc_metrics(adata)
    adata = compute_qc_metrics(adata)
    dataset_row, columns = extract_metrics(adata)


    ### Adding onto the end of new df
    saved_df = pd.read_csv(save_path, index_col = 0)
    new_df = pd.DataFrame.from_dict(dataset_row, columns = columns, 
                                    orient='index') 
    print(f"Adding entry < {new_df.index} to dataset")

    ### TESTING FOR DUPLICATED COLUMNS
    duplicate_columns = new_df.columns[new_df.columns.duplicated()]
    # Display results
    if duplicate_columns.any():
        print(f"WARNING: Duplicate columns in {new_df.index}:")
        print(duplicate_columns)

    common_columns = saved_df.columns.intersection(new_df.columns)

    # print(f"Old Columns: {saved_df.columns} \n")
    # print(f"New Columns: {new_df.columns} \n")
    print(f"Common Columns: {common_columns} \n")
    
    # Select only the common columns from both DataFrames
    saved_df_common = saved_df[common_columns]
    new_df_common = new_df[common_columns]

    # Append df2 to df1
    result = pd.concat([saved_df_common, new_df_common], join="inner")
    result.to_csv(save_path, index= True)

    print(f"Saved to disk at path: {save_path}")




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




# Main function to parse the --all flag
def main():
    parser = argparse.ArgumentParser(description="Run QC pipeline for AnnData object.")
    parser.add_argument('--all', action='store_true', help='Run QC on all stored datasets')
    parser.add_argument('--path', action='store', help='Save path for output file')

    args = parser.parse_args()

    if args.path:
        save_path = args.path
    else:
        save_path = "outputs/characteristics/characteristics.csv"

    if args.all:
        try:
            adatas = load_datasets()
            first = adatas[0]
            first = setup_qc_metrics(first)
            first = compute_qc_metrics(first)
            dataset_row, columns = extract_metrics(first)
            characteristics_df = pd.DataFrame.from_dict(dataset_row, columns = columns, orient='index')
            characteristics_df.to_csv(save_path, index= True)
        except:
            print(f"Loading dataset failed, or loading the first adata failed.")

        for adata in adatas[1:]:
            try:
                run_all_qc(adata, save_path = save_path)
            except:
                print(f"{adata.uns['name']} failed. Proceeding with the next")

        # if args.path:
        #     print(f"Saving to path: {args.path} \n")
        #     characteristics_df.to_csv(args.path, index=True)
        # else:
        #     save_path = "outputs/characteristics/characteristics.csv"
        #     print(f"Saving to path: {save_path} \n")
        #     characteristics_df.to_csv(save_path, index=True)
        
if __name__ == "__main__":
    main()