from parser.data_parser import DataParser
import numpy as np
from bayes import Bayes


if __name__ == "__main__":
    parser = DataParser("dataset/iris.csv")
    parser.download()
    carac, species = parser.parse()
    X_train, X_test, y_train, y_test = parser.split_data(carac, species)
    bayes = Bayes()

    grouped_data = bayes.group(X_train, y_train)
    summary = bayes.summerize(grouped_data)
    priors = bayes.prepare_priors(grouped_data, y_train)
    predictions = bayes.make_prediction(X_test, summary, priors)
    acc = bayes.accuracy_metric(y_test, predictions)

    print(f"Accuracy: {acc * 100:.2f}%")
