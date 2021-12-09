import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, \
    recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import logistic_regression
import util
from constants import DataDir
from constants import MODE

np.set_printoptions(threshold=np.inf)
PATH_OUTPUT = os.getcwd() + "/output/random_forest/"


def main():
    util.log("Reading data")
    training = util.get_clean_dataframe_from_file(DataDir.all_tables)
    training = util.convert_input_column_type(training)

    # print out information about the dataset
    temp = training
    util.log(temp['attack_cat'].value_counts())
    util.log('Unreduced (benign included) label code mapping')
    util.log(dict(zip(temp['attack_cat'].cat.codes, temp['attack_cat'])))
    temp = temp[temp['attack_cat'] != 'benign']
    util.log('Reduced (no benign) label code mapping')
    util.log(dict(zip(temp['attack_cat'].cat.codes, temp['attack_cat'])))

    # Binary
    # Raw training data
    x_train, y_train = util.get_input_output_rf(training, class_type='binary')
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_random_forest(x_train, y_train, x_test, y_test, MODE.binary)

    # PCA
    x_train, y_train = util.get_input_output_rf(training, class_type='binary')
    pca = PCA(0.99, random_state=42)
    x_train = pca.fit_transform(x_train)
    util.log(f'The number of principal components is: {x_train.shape[0]}')
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_random_forest(x_train, y_train, x_test, y_test, MODE.binary_PCA)

    # No benign
    # Raw training data WITHOUT benign labels
    x_train, y_train = util.get_input_output_rf(training, class_type='multiclass', benign_include=False)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_random_forest(x_train, y_train, x_test, y_test, MODE.no_benign)

    # PCA WITHOUT benign labels
    x_train, y_train = util.get_input_output_rf(training, class_type='multiclass', benign_include=False)
    pca = PCA(0.99, random_state=42)
    x_train = pca.fit_transform(x_train)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_random_forest(x_train, y_train, x_test, y_test, MODE.no_benign_PCA)


def run_random_forest(x_train,
                      y_train,
                      x_test,
                      y_test,
                      mode):
    clf = RandomForestClassifier(max_depth=None,
                                 n_estimators=150,
                                 random_state=42,
                                 n_jobs=-1)
    util.log("=============================Fitting=============================")
    util.log(f"Current type: {mode.value}")
    clf.fit(x_train, y_train)
    util.log(f"Predicting")
    predict = clf.predict(x_test)

    # Confusion matrix
    util.log("Generating confusion matrix")
    cm = confusion_matrix(y_test, predict)
    display_cm = ConfusionMatrixDisplay(confusion_matrix=cm)
    display_cm.plot()
    plt.title(f'Random Forest {mode.value} Confusion Matrix')
    util.log(f'Random Forest {mode.value} Confusion Matrix')
    util.log(display_cm.confusion_matrix)
    plt.savefig(PATH_OUTPUT + f'Random Forest Confusion Matrix - {mode.value}.png')
    plt.close()

    # scores
    util.log(f"Accuracy - {mode.value}: {accuracy_score(y_test, predict)}")
    # These scores only available to binary
    if MODE.is_binary(mode):
        util.log(f"F1 Score - {mode.value}: {f1_score(y_test, predict)}")
        util.log(f"Precision - {mode.value}: {precision_score(y_test, predict)}")
        util.log(f"Recall - {mode.value}: {recall_score(y_test, predict)}")


if __name__ == '__main__':
    main()
