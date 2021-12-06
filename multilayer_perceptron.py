import os

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, \
    recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.utils.np_utils import to_categorical

import logistic_regression
import model_utils
import util
from constants import DataDir
from constants import MODE

PATH_OUTPUT = os.getcwd() + "/output/multilayer_perceptron/"
np.set_printoptions(threshold=np.inf)


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
    x_train, y_train = util.get_input_output(training, class_type='binary')
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_multilayer_perceptron(x_train, y_train, x_test, y_test, MODE.binary, class_type='binary')

    # Top features from logistic regression
    x_train, y_train = util.get_input_output(training, class_type='binary')
    top_features = util.run_logistic_regression_feature_reduction(x_train, y_train, x_test, y_test, MODE.binary)
    x_train = x_train[[x[0] for x in top_features]]
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_multilayer_perceptron(x_train, y_train, x_test, y_test, MODE.binary_reduced, class_type='binary')

    # PCA
    x_train, y_train = util.get_input_output(training, class_type='binary')
    feature_scale = StandardScaler()  # scale the data for use with PCA
    x_train = feature_scale.fit_transform(x_train)
    pca = PCA(0.99, random_state=42)
    x_train = pca.fit_transform(x_train)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_multilayer_perceptron(x_train, y_train, x_test, y_test, MODE.binary_PCA, class_type='binary')

    # Multi-class
    # Raw training data
    x_train, y_train = util.get_input_output(training, class_type='multiclass')
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_multilayer_perceptron(x_train, y_train, x_test, y_test, MODE.multi, class_type='multiclass')

    # Top features from above
    x_train, y_train = util.get_input_output(training, class_type='multiclass')
    top_features = util.run_logistic_regression_feature_reduction(x_train, y_train, x_test, y_test, MODE.multi)
    x_train = x_train[[x[0] for x in top_features]]
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_multilayer_perceptron(x_train, y_train, x_test, y_test, MODE.multi_reduced, class_type='multiclass')

    # PCA
    x_train, y_train = util.get_input_output(training, class_type='multiclass')
    feature_scale = StandardScaler()  # scale the data for use with PCA
    x_train = feature_scale.fit_transform(x_train)
    pca = PCA(0.99, random_state=42)
    x_train = pca.fit_transform(x_train)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_multilayer_perceptron(x_train, y_train, x_test, y_test, MODE.multi_PCA, class_type='multiclass')

    # No benign
    # Raw training data WITHOUT benign labels
    x_train, y_train = util.get_input_output(training, class_type='multiclass', benign_include=False)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_multilayer_perceptron(x_train, y_train, x_test, y_test, MODE.no_benign, class_type='multiclass')

    # Top features from above WITHOUT benign labels
    x_train, y_train = util.get_input_output(training, class_type='multiclass', benign_include=False)
    top_features = util.run_logistic_regression_feature_reduction(x_train, y_train, x_test, y_test, MODE.no_benign)
    x_train = x_train[[x[0] for x in top_features]]
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_multilayer_perceptron(x_train, y_train, x_test, y_test, MODE.no_benign_reduced, class_type='multiclass')

    # PCA WITHOUT benign labels
    x_train, y_train = util.get_input_output(training, class_type='multiclass', benign_include=False)
    feature_scale = StandardScaler()
    x_train = feature_scale.fit_transform(x_train)
    pca = PCA(0.99, random_state=42)
    x_train = pca.fit_transform(x_train)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_multilayer_perceptron(x_train, y_train, x_test, y_test, MODE.no_benign_PCA, class_type='multiclass')


def run_multilayer_perceptron(x_train,
                              y_train,
                              x_test,
                              y_test,
                              mode,
                              class_type):
    if class_type == 'binary':
        clf = model_utils.create_model_binary(x_train.shape[1])
    else:
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        clf = model_utils.create_model_multiclass(x_train.shape[1], y_train.shape[1])

    util.log("=============================Fitting=============================")
    util.log(f"Current type: {mode.value}")
    callback = EarlyStopping(monitor='loss', patience=3)
    clf.fit(
        x_train,
        y_train,
        epochs=200,
        batch_size=1000,
        callbacks=[callback],
        verbose=1
    )

    util.log(f"Predicting")
    predict = clf.predict(x_test)

    if class_type != 'binary':
        predict = np.argmax(predict, axis=1)
        y_test = np.argmax(y_test, axis=1)
    else:
        predict = np.around(predict)

    # Confusion matrix
    util.log("Generating confusion matrix")
    cm = confusion_matrix(y_test, predict)
    display_cm = ConfusionMatrixDisplay(confusion_matrix=cm)
    display_cm.plot()
    plt.title(f'Multi-layer Perceptron {mode.value} Confusion Matrix')
    util.log(f'Multi-layer Perceptron {mode.value} Confusion Matrix')
    util.log(display_cm.confusion_matrix)
    plt.savefig(PATH_OUTPUT + f'Multi-layer Perceptron Confusion Matrix - {mode.value}.png')
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
