import numpy as np
from typing import Optional, Union


class GaussianNaiveBayes:
    def __init__(self):
        self.class_stats: dict[str, dict[str, Union[np.ndarray, float]]] = {}
        self.classes: Optional[np.ndarray] = None

    def __calculate_probability(self, x, mean, stdev):
        """
            Calculates the Gaussian probability density function for a given value, mean, and standard deviation.

        Args:
            x (_type_): The value for which to calculate the probability.
            mean (_type_): The mean of the distribution.
            stdev (_type_): The standard deviation of the distribution.

        Returns:
            _type_: The calculated probability density for the given value, mean, and standard deviation.
        """
        return (1 / (np.sqrt(2 * np.pi) * stdev)) * np.exp(
            -((x - mean) ** 2 / (2 * stdev**2))
        )

    def __group_and_summerize(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Groups the data by class and calculates statistics for each class.

        Args:
            X_train (np.ndarray): The input features, shape (n_samples, n_features).
            y_train (np.ndarray): The target labels, shape (n_samples,).
        """
        self.classes = np.unique(y_train)

        for cls in self.classes:
            X_cls: np.ndarray = X_train[y_train == cls]
            self.class_stats[cls] = {
                "mean": X_cls.mean(axis=0),
                "stdev": X_cls.std(axis=0) + 1e-8,
                "prior": len(X_cls) / len(X_train),
            }

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
            Trains the Gaussian Naive Bayes model by grouping the data by class and calculating statistics for each class.

        Args:
            X_train (np.ndarray): The input features, shape (n_samples, n_features).
            y_train (np.ndarray): The target labels, shape (n_samples,).
        """
        self.__group_and_summerize(X_train, y_train)

    def predict(self, X: np.ndarray):
        """
            Predicts the class labels for the given input features.

        Args:
            X (np.ndarray): The input features, shape (n_samples, n_features).

        Returns:
            _type_: The predicted class labels, shape (n_samples,).
        """
        predictions = []
        for x in X:
            probabilities: dict[str, float] = {}
            if self.classes is not None:
                for cls in self.classes:
                    mean = self.class_stats[cls]["mean"]
                    stdev = self.class_stats[cls]["stdev"]
                    prior = self.class_stats[cls]["prior"]

                    likelihood = np.prod(self.__calculate_probability(x, mean, stdev))
                    probabilities[cls] = likelihood * prior
            predictions.append(max(probabilities, key=probabilities.get))  # type: ignore
        return predictions

    def accuracy_metric(self, y_true, y_pred):
        """
            Calculates the accuracy of the predictions by comparing the true labels with the predicted labels.

        Args:
            y_true (_type_): The true labels.
            y_pred (_type_): The predicted labels.

        Returns:
            _type_: The accuracy of the predictions.
        """
        correct = sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i])
        return correct / len(y_true)
