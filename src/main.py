from parser.data_parser import DataParser
import numpy as np


def group(x, y):
    group_by_species = {}
    for feature, label in zip(x, y):
        if label not in group_by_species:
            group_by_species[label] = []
        group_by_species[label].append(feature)
    return group_by_species


def summerize(x):
    mdata = {}
    for labels, features in x.items():
        mdata[labels] = [
            (np.mean(feature), np.std(feature, ddof=1)) for feature in zip(*features)
        ]
    return mdata


def calculate_probability(x, mean, stdev):
    return (1 / (np.sqrt(2 * np.pi) * stdev)) * np.exp(
        -((x - mean) ** 2 / (2 * stdev**2))
    )


def accuracy_metric(y_true, y_pred):
    correct = sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i])
    return correct / len(y_true)


if __name__ == "__main__":
    parser = DataParser("dataset/iris.csv")
    parser.download()
    carac, species = parser.parse()
    X_train, X_test, y_train, y_test = parser.split_data(carac, species)
    grouped_data = group(X_train, y_train)
    summary = summerize(grouped_data)

    total_samples = len(y_train)
    priors = {}
    for label, features in grouped_data.items():
        priors[label] = len(features) / total_samples

    predictions = []
    for i in range(len(X_test)):
        prob = {}
        for label, features in summary.items():
            prob[label] = priors[label]
            for j in range(len(features)):
                mean, stdev = features[j]
                x = X_test[i][j]
                prob[label] *= calculate_probability(x, mean, stdev)
        best_label, best_prob = None, -1
        for class_value, probability in prob.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        predictions.append(best_label)
    acc = accuracy_metric(y_test, predictions)
    print(f"Accuracy: {acc * 100:.2f}%")
