import numpy as np
from sklearn.model_selection import train_test_split

class NaiveBayes:
    def __init__(self, L):
        #initialize nbc with a value for L
        self.L = L
        # store future prior class probabilities to a dict
        self.class_probabilities = {}
        # store future conditional probabilities to a dict
        self.feature_probabilities = {}
        self.D_categorical = None
        self.Y = None

    def train(self, X, X_dtype, Y, D_categorical=None):
        #load training datasets and convert it to np arrays
        X = np.genfromtxt(X, delimiter=',', dtype=float)
        Y = np.genfromtxt(Y, delimiter=',', dtype=int)
        #load D categorical dataset representing the number of possible values for categorical features
        if D_categorical is not None:
            D_categorical = np.genfromtxt(D_categorical, delimiter=',', dtype=int)

        I, M = X.shape
        classes = np.unique(Y) # extract the unique labels
        num_classes = len(classes)

        self.Y = Y
        self.D_categorical = D_categorical
        # calculate the prior probability for each class
        for cl in classes:
            self.class_probabilities[cl] = (np.sum(Y == cl) + self.L) / (I + self.L * num_classes)

        # calculate conditional probabilities for each feature for categorical data
        if X_dtype == "categorical":
            self.feature_probabilities = {i: {} for i in range(M)}
            for i in range(M):
                for cl in classes:
                    for v in range(D_categorical[i]):
                        occurrences = np.sum((X[:, i] == v) & (Y == cl))
                        # Bayes formula applying Laplace smoothing
                        self.feature_probabilities[i][(cl, v)] = (occurrences + self.L) / (np.sum(Y == cl) + self.L * D_categorical[i])
        # else for continuous data calculate the mean and standard deviation for each feature for a given class
        else:
            self.feature_means = {}
            self.feature_stds = {}
            for i in range(M):
                self.feature_means[i] = {cl: np.mean(X[Y == cl, i]) for cl in classes}
                self.feature_stds[i] = {cl: np.std(X[Y == cl, i]) for cl in classes}

    def predict(self, X, X_dtype):
        J, M = X.shape
        # initialize predictions list to store the predicted class for each sample
        predictions = []

        for j in range(J):
            max_prob = -1
            predicted_class = None

            for c, class_prob in self.class_probabilities.items():
                # initialize a class prior probability
                prob = class_prob

                if X_dtype == 'categorical':
                    # iterate over each feature
                    for i in range(M):
                        feature_value = X[j, i]
                        # calculate conditional prob for the feature given the class using laplace smoothing form and adding L*D to the nominator
                        prob *= self.feature_probabilities[i].get((c, feature_value), self.L / (np.sum(self.Y == c) + self.L * self.D_categorical[i]))
                else:
                    # iterate over each feature of cntinuous data
                    for i in range(M):
                        feature_value = X[j, i]
                        # calculate mean and std statistic and store them
                        mean = self.feature_means[i][c]
                        std = self.feature_stds[i][c]
                        # calculate pdf with the normal distribution formula
                        prob *= (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((feature_value - mean) / std) ** 2)

                if prob > max_prob:
                    max_prob = prob
                    predicted_class = c

            predictions.append(predicted_class)

        return np.array(predictions)

def calculate_accuracy(predictions, actual_labels):
    return np.mean(predictions == actual_labels)

def evaluate_naive_bayes(data_path, data_type, L, test_size, train_size, iterations):
    accuracies_cat, accuracies_cont = [], []

    for _ in range(iterations):
        X = np.genfromtxt(data_path['X'], delimiter=',', dtype=float)
        Y = np.genfromtxt(data_path['Y'], delimiter=',', dtype=int)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, train_size=train_size, random_state=None)

        nbc_cat, nbc_cont = NaiveBayes(L), NaiveBayes(L)
        nbc_cat.train(data_path['X'], data_type, data_path['Y'], data_path['D_categorical'])
        predictions_cat = nbc_cat.predict(X_test, data_type)

        nbc_cont.train(data_path['X'], data_type, data_path['Y'], data_path['D_categorical'])
        predictions_cont = nbc_cont.predict(X_test, data_type)

        accuracies_cat.append(calculate_accuracy(predictions_cat, Y_test))
        accuracies_cont.append(calculate_accuracy(predictions_cont, Y_test))

    average_accuracy_cat = np.mean(accuracies_cat)
    average_accuracy_cont = np.mean(accuracies_cont)

    return average_accuracy_cat, average_accuracy_cont

data_path = {
    'X': 'DatasetB_X_continuous.csv',
    'Y': 'DatasetB_Y.csv',
    'D_categorical': None # or give the path for D_categorical dataset
}

L1 = 1
L1000 = 1000
test_size = 0.25
train_size = 0.75
num_iterations = 100
data_type = 'continuous'

average_accuracy_catL1, average_accuracy_cont = evaluate_naive_bayes(data_path, data_type, L1, test_size, train_size, num_iterations)
average_accuracy_catL1000, average_accuracy_cont = evaluate_naive_bayes(data_path, data_type, L1000, test_size, train_size, num_iterations)

print(f"Average Accuracy for {data_type} Data over {num_iterations} iterations and L = {L1}: {average_accuracy_catL1}")
print(f"Average Accuracy for {data_type} Data over {num_iterations} iterations and L = {L1000}: {average_accuracy_catL1000}")
