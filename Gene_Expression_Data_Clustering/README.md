# Gene Expression Data Clustering Project

For the purposes of this project, data were collected from the following study: 

DOI: 10.1016/j.jaci.2011.10.038 (DataSet Record: GDS4424)

Requirements

    Python 3.x
    Packages:
        pandas
        numpy
        matplotlib
        seaborn
        scikit-learn

## Overview

This project focuses on gene expression data clustering, aiming to investigate molecular networks that are optentially realated to viral-triggeredasthma exacerbations in children by identifying co-regulated gene clusters/gaining insights into biological pathways involved asthma exacerbations.

To achieve this, gene expression data were analyzed for two periods: 

* During acute exacerbations
* During the subsequent recovery phases

## Project Goals

* Analyze gene expression patterns related to asthma exacerbations.
* Cluster genes with similar expression profiles during exacerbation and post-infection conditions.
* Visualize gene expression patterns using heatmaps to identify co-expression clusters.
* Evaluate clustering performance using silhouette scores.


## Key Findings

  * Optimal Variance Threshold:
    A variance threshold of 6.6 yielded the highest silhouette score (0.778), indicating optimal clustering performance.

  * Gene Expression Patterns:
    Clustering revealed that certain genes were upregulated during exacerbation and downregulated post-infection, suggesting involvement in immune responses.

  * Heatmap Visualizations:
    Heatmaps showed distinct gene clusters with high expression levels during exacerbation, providing insights into gene regulation under different conditions.
