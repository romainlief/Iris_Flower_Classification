from parser.data_parser import DataParser
import numpy as np
from bayes import Bayes
from bayes import GaussianNaiveBayes


if __name__ == "__main__":
    parser = DataParser("dataset/iris.csv")
    parser.download()
    carac, species = parser.parse()
    X_train, X_test, y_train, y_test = parser.split_data(carac, species)
        
    gnb = GaussianNaiveBayes()
    gnb.fit(X_train, y_train)
    gnb_predictions = gnb.predict(X_test)
    gnb_acc = gnb.accuracy_metric(y_test, gnb_predictions)
    print(f"Gaussian Naive Bayes Accuracy: {gnb_acc * 100:.2f}%")
