import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import util

from constants import DataDir
from sklearn.model_selection import cross_validate, KFold, train_test_split


def run_mlp_classification(x_train: np.ndarray,
                           y_train: np.ndarray,
                           x_test: np.ndarray,
                           y_test: np.ndarray,
                           class_type: str = 'binary') -> None:
    """
    Runs a multilayer perceptron on the given input and output data.

    :param x_train: Training features
    :param y_train: Training labels
    :param x_test: Testing features
    :param y_test: Testing labels
    :param class_type: The type of class. (binary, multiclass)
    :return: Nothing, a "pure" IO operation.
    """
    mlp_clf = MLPClassifier(hidden_layer_sizes=(50, 50),
                            activation='relu',
                            solver='adam',
                            learning_rate_init=1e-3,
                            early_stopping=True)
    print("Fitting")
    mlp_clf.fit(x_train, y_train)
    print("Predicting")
    score = mlp_clf.score(x_test, y_test)
    print(f"Accuracy - {class_type}: {score}")


if __name__ == '__main__':
    print("Reading data")
    training = util.get_clean_dataframe_from_file(DataDir.all_tables)
    training = util.convert_input_column_type(training)

    # use training data for prediction
    x_train, y_train = util.get_input_output(training, class_type='multiclass')

    # scale the data for use with PCA
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    # apply principal components analysis
    pca = PCA(0.99)
    x_train = pca.fit_transform(x_train)

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25)

    run_mlp_classification(x_train, y_train, x_test, y_test, class_type='multiclass')
