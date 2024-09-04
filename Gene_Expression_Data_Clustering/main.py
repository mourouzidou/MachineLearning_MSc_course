import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score


# --------------------------------------------------
#             Load the Dataset
# --------------------------------------------------

def get_data(filepath):
    return pd.read_csv(filepath, delimiter="\t")


def load_expression_data(filepath):
    expression_df = get_data(filepath)

    # store sample labels according to the pattern post infection, during exacerbation
    # 0 representing during exacerbation condition and
    # 1 representing after infection condition
    labels = np.array([1, 0] * (expression_df.shape[1] // 2))
    expression_df.loc['Condition'] = labels

    return expression_df


# --------------------------------------------------------
#             Data insights - information & Transformation
# --------------------------------------------------------

def get_data_insights(expression_df):
    # Get the minimum and maximum values of the DataFrame
    # to adjust the scale of visualization later
    min_value = expression_df.min().min()
    max_value = expression_df.max().max()
    value_range = max_value - min_value
    median_value = expression_df.median()

    return min_value, max_value, value_range, median_value


def conditions_grouped_df(expression_df):
    # Store the columns with label == 0 and label == 1 separately
    columns_1 = expression_df.columns[expression_df.iloc[-1] == 1]
    columns_0 = expression_df.columns[expression_df.iloc[-1] == 0]

    # Transform the expression_df to group together columns
    # by their condition for better visualization
    return pd.concat([expression_df[columns_0], expression_df[columns_1]], axis=1)


def filtered_var_df(expression_df, threshold):
    # Specify the variance of gene expression levels -
    # remove genes with very low variance
    onlyexp_df = expression_df.drop('Condition')
    range_values = onlyexp_df.var(axis=1)
    range_df = pd.DataFrame(range_values, columns=['Range'])
    filtered_df = onlyexp_df[range_values >= threshold]

    return filtered_df, range_df


# --------------------------------------------------------
#                  K-MEANS CLUSTERING
# --------------------------------------------------------

def perform_kmeans_clustering(data, num_clusters=2):
    k_means = KMeans(num_clusters)
    k_means.fit(data)
    cluster_labels = k_means.labels_
    data['Cluster'] = cluster_labels
    data_sorted = data.sort_values(['Cluster'])
    return data, data_sorted, cluster_labels


def performance(data, labels):
    return silhouette_score(data, labels)


def plot_silhouette_scores(expression_df, min_threshold, max_threshold, step_size):
    silhouette_scores = []
    thresholds = np.arange(min_threshold, max_threshold + step_size, step_size)

    for threshold in thresholds:
        transformed_df = conditions_grouped_df(expression_df)
        filtered_df, range_df = filtered_var_df(transformed_df, threshold)
        clustered_data, clustered_data_sorted, cluster_labels = perform_kmeans_clustering(filtered_df)
        silhouette = performance(clustered_data, cluster_labels)
        silhouette_scores.append(silhouette)

    # figure size and background color
    plt.figure(figsize=(8, 6))
    plt.style.use('seaborn-white')
    ax = plt.gca()
    ax.set_facecolor('white')

    # Plot  silhouette scores
    plt.plot(thresholds, silhouette_scores, color='steelblue', linewidth=2)

    #  axis labels and title
    plt.xlabel('Variance Threshold', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Silhouette Score vs Variance Threshold', fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.5)

    # Add a  line at the maximum silhouette score
    max_score = max(silhouette_scores)
    plt.axhline(y=max_score, color='darkred', linestyle='--')
    plt.text(max_threshold + 0.1, max_score + 0.02, f"Max Score: {max_score:.2f}", color='darkred', fontsize=10)

    # Show the plot
    plt.show()


def create_gene_expression_heatmap(data, min_val, max_val, variance):
    # Create a heatmap
    palette = sns.color_palette("viridis", n_colors=256)
    colors = np.array(palette)
    # norm = plt.Normalize(min_value, max_value)
    trunc_cmap = LinearSegmentedColormap.from_list('truncated_cmap', colors)

    sns.heatmap(data, cmap=trunc_cmap, vmin=min_val, vmax=max_val)
    plt.xlabel('Conditions')
    plt.ylabel('Genes')
    plt.title(f'Gene Expression Heatmap for variance = {variance}')
    plt.show()


expression_df = load_expression_data("expression_matrix.tsv")
min_value, max_value, value_range, median_value = get_data_insights(expression_df)

transformed_df = conditions_grouped_df(expression_df)
filtered_df_5, range_df = filtered_var_df(transformed_df, 5)
filtered_max, range_df_max = filtered_var_df(transformed_df, 6.6)

clustered_data_0, clustered_data_sorted_0, cluster_labels_0 = perform_kmeans_clustering(transformed_df)
clustered_data_5, clustered_data_sorted_5, cluster_labels_5 = perform_kmeans_clustering(filtered_df_5)
clustered_data_max, clustered_data_sorted_max, cluster_labels_max = perform_kmeans_clustering(filtered_max)

silhouette_5 = performance(clustered_data_5, cluster_labels_5)
silhouette_0 = performance(clustered_data_0, cluster_labels_0)
silhouette_max = performance(clustered_data_max, cluster_labels_max)

min_5, max_5, _, _ = get_data_insights(clustered_data_5)
min_max, max_max, _, _ = get_data_insights(clustered_data_max)

print(silhouette_0)  # 0.46556243265431474
print(silhouette_5)  # 0.5888228879219461
print(silhouette_max)  # 0.7776890015125885

plot_silhouette_scores(expression_df, min_threshold=6, max_threshold=7, step_size=0.1)
create_gene_expression_heatmap(clustered_data_sorted_0, min_value, max_value, 0)
create_gene_expression_heatmap(clustered_data_sorted_5, min_5, max_5, 5)
create_gene_expression_heatmap(clustered_data_sorted_max, min_max, max_max, 6.6)



