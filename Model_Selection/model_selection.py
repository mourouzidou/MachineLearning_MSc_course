import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample

# read the .csv dataset as a pandas dataframe and specify that there is no headeρ row
df = pd.read_csv("Dataset6.A_XY.csv", header=None)

# define the data (X) and the target categorical variable (Y)
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

# define lists to store categorical and continuous data as specified by the data exploration step
# (exclude the last feature with index 50 as it is the target Y variable)
categorical_features = [1, 4, 5, 7, 8, 9, 10, 11, 12, 15, 16,
                        19, 22, 24, 25, 28, 29, 30, 34, 35, 37,
                        40, 41, 44, 47, 48, 49]
continuous_features = [0, 2, 3, 6, 13, 14, 17, 18, 20, 21,
                       23, 26, 27, 31, 32, 33, 36, 38, 39,
                       42, 43, 45, 46]

# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False)
X_categorical = encoder.fit_transform(df.iloc[:, categorical_features])
X_continuous = df.iloc[:, continuous_features].to_numpy()
X = np.hstack((X_categorical, X_continuous))
y = df.iloc[:, -1]
y_binary = y.replace({1.0: 0, 2.0: 1})

# define the list of desired hyperparameter and model configurations

configurations = [
    {'classifier': LogisticRegression,
     'params': {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2', None], 'max_iter': [1000]}},
    {'classifier': KNeighborsClassifier,
     'params': {'n_neighbors': [10, 47, 60], 'weights': ['uniform', 'distance']}},
    {'classifier': RandomForestClassifier,
     'params': {'n_estimators': [100, 500, 1000], 'max_depth': [5, 10, None]}}
]


def create_folds(data, k):
    # initialize a list of k lists - one to store each fold
    folds = [[] for _ in range(k)]
    sorted_data = data.sort_values(data.columns[-1])  # sort rows by Y
    target = sorted_data.iloc[:, -1]

    # count occurrences of each class and split indices in two arrays according to each class
    class_counts = target.value_counts()
    class_indices = {cls: np.where(target == cls)[0] for cls in class_counts.index}

    # iterate through the class dictionaries
    for cls, indices in class_indices.items():
        # randomize data points in each class by shuffling them
        # and split indices to k parts
        np.random.shuffle(indices)
        class_folds = np.array_split(indices, k)
        # add indices to respective fold
        for i in range(k):
            folds[i].extend(class_folds[i])

    # shuffle each fold to mix classes
    for fold in folds:
        np.random.shuffle(fold)
    return folds


def run_model(train_data, test_data, configuration):
    classifier = configuration['classifier'](**configuration['params'])
    # train classifiers using x_train and y_train inputs
    classifier.fit(train_data[0], train_data[1])
    # make predictions on test set using predict_proba to get the probability estimates for ecah class
    predictions = classifier.predict_proba(test_data[0])[:, 1]
    # compute and return roc-auc score
    return roc_auc_score(test_data[1], predictions)


def performance_estimation(data, validation_indices, configurations):
    performances = []  # initialize a list to store mean performance of each configuration
    for config in configurations:  # iterate over each configuration
        # calculate the average roc auc score for current configuration
        performances.append(np.mean([
            run_model((data.iloc[train_idx, :-1], data.iloc[train_idx, -1]),  # define train data
                      (data.iloc[test_idx, :-1], data.iloc[test_idx, -1]),  # define test data
                      config)  # the current model configuration
            # generate train and test indices for each fold
            for fold in validation_indices
            # bboolean indices: 'train_idx' for training data, 'test_idx' for testing data
            for train_idx, test_idx in [(~data.index.isin(fold), data.index.isin(fold))]
        ]))
    # return list for average performances of each configuration
    return performances


def best_model(performances, configurations):
    # get the index of the highest performance score
    best_index = np.argmax(performances)
    # return the configuration that has the highest performance score
    return configurations[best_index]


# _____________________________________________________________
#           run the analysis on given dataset
# _____________________________________________________________


# Create folds
folds = create_folds(df, 5)

# Generate a list with all possible configurations
expanded_configurations = []
for config in configurations:
    for param_set in ParameterGrid(config['params']):
        expanded_configurations.append({'classifier': config['classifier'], 'params': param_set})

performances = performance_estimation(df, folds, expanded_configurations)  # estimate performance for all configs
cross_validation_performance = max(performances)  # store the best models configuration's performance
best_config = best_model(performances, expanded_configurations)  # select best model

import matplotlib.pyplot as plt

# Classifier types for coloring and legend
classifier_types = {LogisticRegression: 'Logistic Regression',
                    KNeighborsClassifier: 'KNN',
                    RandomForestClassifier: 'Random Forest'}

colors = plt.cm.viridis(np.linspace(0, 0.76, len(classifier_types)))
classifier_colors = {classifier: color for classifier, color in
                     zip(classifier_types.keys(), colors)}  # map classifiers to colors

# generate labels and colors for the configurations
config_labels = []
config_colors = []
for i, config in enumerate(expanded_configurations):
    classifier = config['classifier']
    config_labels.append(f"Config {i + 1}")
    config_colors.append(classifier_colors[classifier])
plt.figure(figsize=(15, 6))
bars = plt.bar(config_labels, performances, color=config_colors)

# highlight the best configuration
best_config_index = np.argmax(performances)
bars[best_config_index].set_edgecolor('black')
bars[best_config_index].set_linewidth(3)

# add an arrow and text for the best configuration
best_bar = bars[best_config_index]
plt.annotate('Best', xy=(best_bar.get_x() + best_bar.get_width() / 2, best_bar.get_height()),
             xytext=(best_bar.get_x() + best_bar.get_width(), best_bar.get_height() * 1.1),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
             ha='center')

# create a legend for the classifiers
legend_labels = [classifier_types[cls] for cls in classifier_types]
legend_colors = [classifier_colors[cls] for cls in classifier_types]
plt.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=color) for color in legend_colors],
           labels=legend_labels, title="Classifier Types", loc="upper center", bbox_to_anchor=(1.05, 1))

# adding configuration text to the plot
for bar, config in zip(bars, expanded_configurations):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval,
             str(config['params']),
             ha='center', va='top',
             rotation=90, fontsize=12, color='white')

plt.xlabel('Configurations')
plt.ylabel('ROC AUC Score')
plt.title('Performance of Each Configuration')
plt.xticks(rotation=45)
plt.show()

# _____________________________________________________
#         train final model on complete dataset
# _____________________________________________________

best_classifier = best_config['classifier']
best_params = best_config['params']

final_model = best_classifier(**best_params)

# Train the model on the entire original dataset based on the best configurations
final_model.fit(X, y_binary)

# _____________________________________________________
#        PART 3 - Out of Sample Performance
#                 ROC CURVE
# _____________________________________________________


# load the file with the test dataset as a pd dataframe
test_df = pd.read_csv("Dataset6.B_XY.csv", header=None)

# Separate features and target variable for the test set as before
X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1].replace({1.0: 0, 2.0: 1})  # replace 1, 2 categories with 0, 1 -> boolean binary format

# specify data types according to the ones of the previous train dataset
X_test_categorical = encoder.transform(X_test.iloc[:, categorical_features])
X_test_continuous = X_test.iloc[:, continuous_features].to_numpy()
X_test_transformed = np.hstack((X_test_categorical, X_test_continuous))

# use the best final model we trained before on the entire original dataset
# and predict probabilities of the test set
y_pred_proba = final_model.predict_proba(X_test_transformed)[:, 1]

# compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
out_of_sample_performance = auc(fpr, tpr)
print("cross validation performance  = ", cross_validation_performance)
print("out of sample performance = ", out_of_sample_performance)
performace_difference = cross_validation_performance = out_of_sample_performance
print("cross validation - out of sample = ", performace_difference)

# get the majority class to "build" the naive classifier
majority_class = y_test.mode()[0]
y_pred_naive = np.full(len(y_test), majority_class)
fpr_naive, tpr_naive, _ = roc_curve(y_test, y_pred_naive)
roc_auc_naive = auc(fpr_naive, tpr_naive)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color=plt.cm.viridis(0.4), lw=2,
         label=f'Best Model ROC curve (area = {cross_validation_performance:.2f})')
# plot ROC curve (diagonal) for a random classifier
plt.plot([0, 1], [0, 1], color=plt.cm.viridis(0.6), lw=2, linestyle='--', label='Random Classifier (area = 0.5)')

plt.scatter(fpr_naive[1], tpr_naive[1], color=plt.cm.viridis(0.8), label='Naive Classifier (point)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.show()


# _________________________
#       PART 4
#  Bootstrap sampling
# _________________________

n_samples = 1000

bootstrap_performances = []

for _ in range(n_samples):
    X_test_bootstrap, y_test_bootstrap = resample(X_test_transformed, y_test)
    roc_auc = roc_auc_score(y_test_bootstrap, final_model.predict_proba(X_test_bootstrap)[:, 1])
    bootstrap_performances.append(roc_auc)
lower_bound, upper_bound = np.percentile(bootstrap_performances, [2.5, 97.5])

plt.hist(bootstrap_performances, bins=30, color= plt.cm.viridis(0.45))
plt.axvline(lower_bound, color= plt.cm.viridis(0.6), linestyle='--', linewidth=2)
plt.axvline(upper_bound, color= plt.cm.viridis(0.6), linestyle='--', linewidth=2)
plt.axvline(out_of_sample_performance, color= plt.cm.viridis(0.9), linestyle='--', linewidth=2)
plt.title('Bootstrap ROC-AUC scores for 95% Confidence Interval')
plt.xlabel('ROC AUC Score')
plt.ylabel('Frequency')
plt.legend(['Lower Bound', 'Upper Bound', 'Original Performance'])
plt.show()