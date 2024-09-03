import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Create a pandas dataframe to store the dataset
df = pd.read_csv("Dataset6.A_XY.csv", header=None)


num_features = len(df.columns)

# check which features are categorical and which continuous
num_cols_per_figure = 6
num_rows_per_figure = 2


# Plot histograms for each feature in separate figures
# for fig_num in range(num_features):
#     # Calculate the range of features for the current figure
#     start_idx = fig_num * num_cols_per_figure * num_rows_per_figure
#     end_idx = min((fig_num + 1) * num_cols_per_figure * num_rows_per_figure, num_features)
#     num_cols = min(end_idx - start_idx, num_cols_per_figure)
#     num_rows = min(num_rows_per_figure, math.ceil((end_idx - start_idx) / num_cols_per_figure))
#
#     # Calculate the dynamic figsize based on the number of rows and columns
#     figsize = (num_cols * 6, num_rows * 6)
#
#     fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize)
#     axes = axes.flatten()
#
#     # Plot histograms for the current set of features
#     for i, col_idx in enumerate(range(start_idx, end_idx)):
#         df[df.columns[col_idx]].hist(ax=axes[i], bins=20, color=cm.viridis(0.90), alpha=0.7)
#         axes[i].set_title(f'Feature {df.columns[col_idx] + 1}', fontsize = 27)
#
#         # Add unique values as a comment inside the plot
#         unique_values_comment = f'Unique : {df[df.columns[col_idx]].nunique()}'
#         axes[i].text(0.5, 0.4, unique_values_comment, fontsize= 32, ha='center', va='top',
#                      color = cm.viridis(0.13), transform=axes[i].transAxes, )
#         variance_text = f'Variance: {df[df.columns[col_idx]].var():.2f}'
#         axes[i].text(0.5, 0.3, variance_text, fontsize=32, ha='center', va='top',
#                      color=cm.viridis(0.13), transform=axes[i].transAxes)
#
#     # plt.tight_layout()
#     # fig.savefig(f'figure_{fig_num + 1}.png')
#     # plt.show()


# Set seaborn style
sns.set(style="whitegrid")

# Compute the unique values number per feature
nuniques = [(col + 1, df[col].nunique()) for col in df.columns]

# Sort the features by their unique values number to pick a proper unique values threshold
sorted_uniques = sorted(nuniques, key=lambda x: x[1])

# Create a list to represent the distribution shape given by the plots above
is_bell = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
           1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0]

# Get unique thresholds
thresholds = sorted(set([t[1] for t in sorted_uniques]))

# Set up the grid
num_rows = 5
num_cols = 3
# # _____Create a grid of subplots______

# fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 10))
# axes = axes.flatten()
#
# # Iterate over unique value thresholds
# for i, threshold_value in enumerate(thresholds):
#     # Scatter plot with points colored based on is_bell values and classified_as
#     classified_as = [0 if unique_values <= threshold_value else 1 for _, unique_values in sorted_uniques]
#
#     # Calculate the agreement percentage
#     agreement_percentage = sum(1 for c, is_bell_value in zip(classified_as, is_bell) if c == is_bell_value) / len(is_bell) * 100
#
#     sns.scatterplot(x=[unique_values for _, unique_values in sorted_uniques], y=is_bell,
#                     hue=classified_as, ax=axes[i], palette='viridis', legend='full', s=50)
#     axes[i].set_title(f'Threshold: {threshold_value}', fontsize=19)
#     axes[i].set_xlabel('Number of Unique Values')
#     axes[i].set_ylabel('Bell Distribution ')
#
#
#     # Add text annotation for agreement percentage
#     text = f'Agreement: {agreement_percentage:.2f}%'
#     axes[i].text(0.5, 0.6, text, horizontalalignment='center', verticalalignment='center',
#                  transform=axes[i].transAxes, fontsize=13)
#
#     axes[i].set_ylim(-0.2, 1.2)
#     axes[i].set_yticks([0, 0.5, 1])
#     axes[i].set_yticklabels(['No', '', 'Yes'])
#     fig.suptitle('Thresholds and Distribution Shape', fontsize=22, y=0.99)
# # Adjust layout
# plt.tight_layout()

# plt.savefig("thresholds.png")
# plt.show()

optimal_threshold = 9

# define continuous and categorical features based on optimal_threshold
continuous_features = [cont[0]-1 for cont in nuniques if cont[1] >= optimal_threshold]
print(f"continuous : {continuous_features}")
categorical_features = [cat[0]-1 for cat in nuniques if cat[1] < optimal_threshold]
print(f"categorical : {categorical_features}")


#______________________________________________
# define correlation of features with target Y
#______________________________________________


#  ~   for continuous features
pos_correlation_thresh = 0.
neg_correlation_thresh = 0
cont_correlationn_withY = df[continuous_features].corrwith(df.iloc[:, -1])
# print(cont_correlationn_withY)
high_corr_positive = cont_correlationn_withY[cont_correlationn_withY > pos_correlation_thresh]
high_corr_negative = cont_correlationn_withY[cont_correlationn_withY < neg_correlation_thresh]

# print("pos: ", high_corr_positive)
# print("neg: ", high_corr_negative)


# #  ~   for categorical features
categorical_vals = [df.iloc[:, i] for i in categorical_features]

# Assuming 'categorical_features' and 'df' are defined as mentioned in your question

dependent_features = []

for i in categorical_features:
    contingency_table = pd.crosstab(df.iloc[:, i], df.iloc[:, -1])
    chi2, p, _, _ = chi2_contingency(contingency_table)

    # set a significance level =0.05
    significance_level = 0.05

    if p < significance_level:
        dependent_features.append(i)

print("dependent : ", dependent_features)


# so , we can retrieve independent features as the difference of the two lists:
independent_categorical = [cat for cat in categorical_features if cat not in dependent_features]


print("Independent Categorical Features:", independent_categorical)
# plt.figure(figsize=(15, 10))
# for i, feature_index in enumerate(dependent_features, 1):
#     plt.subplot(3, 4, i)
#     sns.histplot(x=df.iloc[:, feature_index], hue=df.iloc[:, -1], palette=sns.color_palette("viridis", n_colors=2), kde=True)
#     plt.xlabel(f'Feature with index = {df.columns[feature_index]}')
#     plt.title(f' Feature {feature_index+1} and Y class')
#
# plt.suptitle("Count Plots for Dependent Categorical Features", y=0.98, fontsize=16)
# plt.tight_layout()
#
# plt.savefig('dependent.png')
# plt.show()
#
# # Visualization for independent categorical features
# plt.figure(figsize=(15, 10))
# for i, feature_index in enumerate(independent_categorical, 1):
#     plt.subplot(4, 4, i)
#     sns.histplot(x=df.iloc[:, feature_index], hue=df.iloc[:, -1], palette=sns.color_palette('viridis', n_colors=2), kde=True)
#     plt.xlabel(f'Feature with index = {df.columns[feature_index]}')
#     plt.title(f' Feature {feature_index+1} and Y class')
#
# plt.suptitle("Count Plots for Independent Categorical Features", y=0.98, fontsize=16)
# plt.tight_layout()
# plt.savefig("independent.png")
# plt.show()


#_____selected features_______

# define a list for categorical and another one for continuous
# features to store the indices
selected_cat = [4, 7, 12]
selected_cont = [42, 20]


# get info about class distribution for categorical selected features
# print the class distribution of each feature (%)
for feature in selected_cat:
    class_distribution = df.iloc[:, feature].value_counts(normalize=True) * 100

    print(f'Class Distribution for {feature + 1} :\n{class_distribution}')

# get info about the continuous selected features
# print for eaxh feature mean and std

for feature in selected_cont:
    mean_value = df[feature].mean()
    std = df[feature].std()
    variance = df[feature].var()
    print(f'{feature} - Mean: {mean_value:.2f}, Std: {std:.2f}, Variance: {variance:.2f}')

print("categorical features", categorical_features)
print("continuous features", continuous_features)

















