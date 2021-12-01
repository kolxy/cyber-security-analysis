import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

import util

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from constants import DataDir
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

PATH_OUTPUT = os.getcwd() + "/output/"


def run_random_forest_classification(x_train: np.ndarray,
                                     y_train: np.ndarray,
                                     x_test: np.ndarray,
                                     y_test: np.ndarray,
                                     class_type: str,
                                     reduced: bool) -> None:
    clf = RandomForestClassifier(max_depth=None,
                                 n_estimators=150,
                                 n_jobs=-1)
    print(f"Fitting - reduced? {reduced} - class type? {class_type}")
    clf.fit(x_train, y_train)
    print(f"Predicting - reduced? {reduced} - class type? {class_type}")
    predict = clf.predict(x_test)
    print("Generating confusion matrix")
    cm = confusion_matrix(y_test, predict)
    display_cm = ConfusionMatrixDisplay(confusion_matrix=cm)
    display_cm.plot()
    if reduced:
        plt.savefig(PATH_OUTPUT + f'random_forest_pca_{class_type}_confusion_matrix.png')
    else:
        plt.savefig(PATH_OUTPUT + f'random_forest_no_pca_{class_type}_confusion_matrix.png')

    print(f"Accuracy - {class_type}: {accuracy_score(y_test, predict)}")


if __name__ == '__main__':
    print("Reading data")
    training = util.get_clean_dataframe_from_file(DataDir.all_tables)
    training = util.convert_input_column_type(training)

    print(training['attack_cat'].value_counts())

    # Binary PCA

    # use training data for prediction
    x_train, y_train = util.get_input_output(training, class_type='binary')

    # scale the data for use with PCA
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    # apply principal components analysis
    pca = PCA(0.99)
    x_train = pca.fit_transform(x_train)

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25)
    run_random_forest_classification(x_train, y_train, x_test, y_test, reduced=True, class_type='binary')

    # Multiclass PCA

    # use training data for prediction
    x_train, y_train = util.get_input_output(training, class_type='multiclass')

    # scale the data for use with PCA
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    # apply principal components analysis
    pca = PCA(0.99)
    x_train = pca.fit_transform(x_train)

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25)
    run_random_forest_classification(x_train, y_train, x_test, y_test, reduced=True, class_type='multiclass')

    # Binary No-PCA

    # use training data for prediction
    x_train, y_train = util.get_input_output(training, class_type='binary')

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25)
    run_random_forest_classification(x_train, y_train, x_test, y_test, reduced=False, class_type='binary')

    # Multiclass No-PCA

    # use training data for prediction
    x_train, y_train = util.get_input_output(training, class_type='multiclass')

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25)
    run_random_forest_classification(x_train, y_train, x_test, y_test, reduced=False, class_type='multiclass')

