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

## Overview - Abstract

The aim of this study is to investigate the molecular networks underlying
viral-triggered asthma exacerbations in children. By analyzing gene expression
data obtained during acute exacerbations and the subsequent recovery phase,
which occurs within 7 to 14 days, we seek to elucidate the mechanisms involved
in these exacerbations. Additionally, the construction of coexpression networks
will contribute to a better understanding of the molecular mechanisms
underlying asthma exacerbations and aid in the development of improved
treatment strategies. Clustering algorithms will be employed to group genes
with similar expression patterns in order to identify co-regulated gene clusters
and gain insights into the biological pathways involved in asthma exacerbations

To achieve this, gene expression data were analyzed for two periods: 

* During acute exacerbations
* During the subsequent recovery phases

## Project Goals

* Analyze gene expression patterns related to asthma exacerbations.
* Cluster genes with similar expression profiles during exacerbation and post-infection conditions.
* Visualize gene expression patterns using heatmaps to identify co-expression clusters.
* Evaluate clustering performance using silhouette scores.

## Methodology

* Data extraction and cleaning
  with `GEOquery` R package
  created `expression_matrix.tsv`

* Data exploration and preprocessing

* Clustering with K-means
     * K-Means clustering: applied `KMeans` class from the `scikit-learn` library. This clustering approach groups data points based on their proximity to the nearest cluster centroid. Initially, two clusters are specified to differentiate between two specific conditions in the gene expression data.
    *  Functions Utilized:

      perform_kmeans_clustering(data, num_clusters=2) 
        * Perform the clustering on the provided data.
        * Add the cluster labels back to the original DataFrame.
        * Sort data by cluster labels for better visualization.
        * Output: Original DataFrame with cluster labels, a sorted DataFrame, and an array of cluster labels.

      performance(data, labels):

        Calculate the silhouette score, which is a metric that evaluates the separation between clusters.
        Output: Silhouette score.

      plot_silhouette_scores(expression_df, min_threshold, max_threshold, step_size):

        Visualize the relationship between various variance thresholds and their corresponding silhouette scores.
        Iterates over a range of variance thresholds, filters data, performs K-means clustering, and calculates silhouette scores.
        Output: Plot of silhouette scores against variance thresholds.
      

  The graph titledillustrates the relationship between different variance thresholds applied to gene expression data and the resulting silhouette scores. These scores measure the quality of clustering, where higher scores indicate better-defined clusters. The optimal silhouette score of 0.778 is achieved at a variance threshold of 6.6. This value is marked on the graph with a dashed red line, indicating the highest cluster quality observed in the tested range.

![image](https://github.com/user-attachments/assets/f58c132b-8bbf-45e6-93e6-e1099da737f3)


   

## Key Findings

#### Figure 2 - Original Dataset (Variance = 0)
This heatmap shows the gene expression data without any variance filtering. All genes are included, leading to a broad representation of expression levels across conditions.

Observations:

    The samples are sorted by condition, exacerbation (left) and post-infection (right), highlighting the gene expression changes associated with these states.
    Higher expression levels (yellowish tones) are more prevalent in the left sections, suggesting that certain genes are upregulated during exacerbation.
    The presence of bluish tones, indicating lower expression, is more scattered but notably concentrated in some parts, especially on the right, which could be linked to the post-infection recovery phase.
    The general stability in expression patterns between conditions suggests that while certain genes show differential expression, the overall gene activity might be consistent across conditions.
  
![image](https://github.com/user-attachments/assets/7dc65cd0-cb2a-4f34-aae1-b5160456d862)


#### Figure 3 - Subset with Variance = 5
With a moderate variance threshold, this heatmap filters out genes with lower variance, focusing on those with more pronounced expression changes.

Observations:

        The distinction between high (left, exacerbation) and low (right, post-infection) expression levels becomes more apparent, emphasizing the biological impact of these conditions on gene regulation.
        This focused view allows for clearer identification of genes that significantly react to infection, potentially useful in pinpointing targets for further biological or therapeutic studies.

![image](https://github.com/user-attachments/assets/4c10ff6c-6876-499a-b70a-a9a109f0a3af)

Figure 4 - Subset for Max Silhouette Score (Variance = 6.6)
This heatmap is filtered at the variance threshold that yielded the highest silhouette score, suggesting optimal clustering and separation of gene expression patterns.

Observations:

    The differentiation between the two conditions is at its most distinct here. The clear division between higher and lower expression areas underscores key regulatory changes during exacerbation versus the recovery phase.
    This visualization could be critical for identifying which genes are most crucial in the response to infection and how they behave differently when the condition shifts from exacerbation to recovery.

 ![image](https://github.com/user-attachments/assets/4da58d9a-7532-4e65-9a3c-f478551f8fc9)


 ## Conclusion - Discussion based on Gene EXpression Heatmap Analysis


The analysis of the gene expression heatmap with a variance threshold of 0 has provided clear insights into the behavior of genes under different conditions of viral infection. Notably, the model effectively distinguishes between genes whose expression is influenced by the infectious condition.

Cluster 2: This cluster includes genes that are upregulated during the infection phase. These genes are likely involved in critical biological processes such as immune responses. The higher expression values, indicated by more yellowish tones in this cluster during the exacerbation condition, suggest active engagement in the body’s defense mechanisms against the infection.

Cluster 1: In contrast, this cluster comprises genes whose expression appears to be unaffected by the viral infection. The consistent expression patterns across both conditions (during exacerbation and 1-2 weeks after infection) in this cluster indicate that these genes maintain regular functions that are not directly modified by the infectious state.

Further Studies: It is beneficial to focus subsequent research efforts on the genes within Cluster 2. Determining the specific biological pathways these genes are involved in can shed light on the molecular mechanisms triggered by picornavirus-induced exacerbations. Understanding these pathways is crucial for developing targeted therapies and interventions aimed at mitigating the impact of such infections.

