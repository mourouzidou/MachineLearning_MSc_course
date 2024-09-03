import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


# ____________________________________________________
#                  PART A
# ____________________________________________________

def TrainRF(X, Y, n_trees, min_samples_leaf):
    max_features = int(np.floor(np.sqrt(X.shape[1])))
    trees = []

    for i in range(n_trees):
        indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
        X_sampled = X[indices, :]
        Y_sampled = Y[indices]

        tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, max_features=max_features)
        tree.fit(X_sampled, Y_sampled)
        trees.append(tree)

    model = {'trees': trees, 'min_samples_leaf': min_samples_leaf, 'max_features': max_features}
    return model


def PredictRF(model, X):
    predictions = np.zeros((X.shape[0], len(model['trees'])))
    for i, tree in enumerate(model['trees']):
        predictions[:, i] = tree.predict(X)

    final_predictions = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=predictions)
    return final_predictions


# ____________________________________________
#           PART B
# ____________________________________________
def calculate_accuracy(tree, X, Y):
    tree_predicts = tree.predict(X)
    accuracy = np.sum(tree_predicts == Y) / len(Y)
    return accuracy * 100


viridis_cmap = cm.get_cmap('viridis')


def plot_accuracy_histogram(ax, individual_accuracies, rf_accuracy, title):

    colors = [viridis_cmap(0.4), viridis_cmap(0.7), viridis_cmap(1)]
    ax.hist(individual_accuracies, color=colors[0], label="Individual Trees", )
    ax.axvline(np.mean(individual_accuracies), color=colors[2], linestyle='dashed', label='Mean Individual Accuracy',
               linewidth=2)
    ax.axvline(rf_accuracy, color=colors[1], linestyle='--', label="Random Forest's Accuracy", linewidth=2)
    ax.set_xlabel("Accuracy (%)", weight="medium", size="x-large")
    ax.set_ylabel('Number of Trees', weight="medium", size="x-large")
    ax.set_title(title)
    ax.legend()


# Load the csv dataset as a pd dataframe
data = pd.read_csv("Dataset5_XY.csv")
X_house_data = np.array(data.iloc[:, :-1])
Y_house_valuable = np.array(data.iloc[:, -1])

# split the data into training 70% training 30% testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_house_data, Y_house_valuable, test_size=0.3, random_state=42)
num_trees = 100

min_samples_leaf_values = [1, 10]  # Add more values as needed

# Create subplots
fig, axes = plt.subplots(1, len(min_samples_leaf_values), figsize=(15, 5))

for i, min_samples_leaf in enumerate(min_samples_leaf_values):
    model = TrainRF(X_train, Y_train, num_trees, min_samples_leaf)
    predictions = PredictRF(model, X_test)
    rf_accuracy = np.sum(predictions == Y_test) * 100 / len(Y_test)

    individual_accuracies = [calculate_accuracy(tree, X_test, Y_test) for tree in model['trees']]

    plot_accuracy_histogram(axes[i], individual_accuracies, rf_accuracy,
                            f"Accuracy of {num_trees} Decision Trees vs Random Forest's Accuracy \n min_sample_leaf = {min_samples_leaf}")

plt.tight_layout()
# plt.show()

# ______________________________________________
#           BONUS
# ______________________________________________

num_trees = 1000
# generate individual decision trees and calculate their accuracies
individual_accuracies = []
for _ in range(num_trees):
    tree = DecisionTreeClassifier()
    tree.fit(X_train, Y_train)
    predictions = tree.predict(X_test)
    accuracy = np.sum(predictions == Y_test) / len(Y_test) * 100
    individual_accuracies.append(accuracy)

# train random forest
rf_model = TrainRF(X_train, Y_train, num_trees, 1)
# store the predicted classes
rf_predictions = PredictRF(rf_model, X_test)
# calculate the random forest's accuracy
rf_accuracy = np.sum(rf_predictions == Y_test) / len(Y_test) * 100


# Calculate probability of achieving similar or better accuracy with a single decision tree
probability = len([acc for acc in individual_accuracies if acc >= rf_accuracy]) / num_trees
print(f"Probability of achieving similar or better accuracy with a single decision tree: {probability:.3f}")

# Visualize distribution of individual decision tree accuracies
viridis_cmap = cm.get_cmap('viridis')
plt.hist(individual_accuracies, color=viridis_cmap(0.4), label="Individual Trees", alpha=0.7, bins=5)
plt.axvline(np.mean(individual_accuracies), color=viridis_cmap(1), linestyle='dashed', label='Mean Individual Accuracy',
            linewidth=2)
plt.axvline(rf_accuracy, color=viridis_cmap(0.7), linestyle='--', label="Random Forest's Accuracy", linewidth=2)
plt.xlabel("Accuracy (%)")
plt.ylabel('Number of Trees')
plt.title(f"Accuracy Distribution of {num_trees} Individual Decision Trees against Random Forest")
plt.legend()
# plt.show()



## repeat 100 times the procedure to retrieve 100 probabilities and observe their distribution
# num_trees = 100
# num_iterations = 100
# all_probabilities = []
#
# for _ in range(num_iterations):
#     # Generate individual decision trees
#     individual_accuracies = []
#     for _ in range(num_trees):
#         tree = DecisionTreeClassifier()
#         tree.fit(X_train, Y_train)
#         predictions = tree.predict(X_test)
#         accuracy = np.sum(predictions == Y_test) / len(Y_test) * 100
#         individual_accuracies.append(accuracy)
#
#     # Calculate Random Forest accuracy
#     rf_model = TrainRF(X_train, Y_train, num_trees, 1)
#     rf_predictions = PredictRF(rf_model, X_test)
#     rf_accuracy = np.sum(rf_predictions == Y_test) / len(Y_test) * 100
#
#     # Calculate probability of achieving similar or better accuracy with a single decision tree
#     probability = len([acc for acc in individual_accuracies if acc >= rf_accuracy]) / num_trees
#     all_probabilities.append(probability)
#
# # Visualize the distribution of probabilities
# plt.hist(all_probabilities, color=viridis_cmap(1), bins=10, edgecolor='black')
# plt.xlabel("Probability")
# plt.ylabel("Frequency")
# plt.title(f"Distribution of Probabilities for Achieving Similar or Better Accuracy \nwith a Single Decision Tree - {num_iterations} iterations")
# plt.show()