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


if __name__ == "__main__":
    parser = DataParser("dataset/iris.csv")
    parser.download()
    carac, species = parser.parse()
    X_train, X_test, y_train, y_test = parser.split_data(carac, species)
    grouped_data = group(X_train, y_train)
    # grouped_test_data = group(X_test, y_test)
    summary = summerize(grouped_data)
    print(summary)
