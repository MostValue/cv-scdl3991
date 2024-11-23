# Developing a Personalised Model for Predicting Optimal Spatial Clustering: A Benchmarking Study

## Project Overview

Spatial transcriptomics (ST) technology allows researchers to map gene expression to spatial locations within tissues. This enables downstream analyses that provide critical insights into processes such as disease progression and development. A key component of ST data analysis is spatial clustering, which leverages both spatial and gene expression information to group cells or regions based on shared characteristics.

## Methodology

We benchmarked clustering performance using:

- 3 clustering methods.
- 5 ST datasets (totaling 19 samples).
- 39 method + hyperparameter combinations.
This resulted in 741 unique clustering outputs, which we evaluated against expert annotations using the Adjusted Rand Index (ARI).

We developed a supervised machine learning model to predict clustering performance based on dataset-specific features and characteristics of the clustering methods and hyperparameters.
The model was trained and validated on the benchmarked clustering results.
