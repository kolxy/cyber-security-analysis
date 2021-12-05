import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, \
    recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import util
from constants import DataDir
from constants import MODE

PATH_OUTPUT = os.getcwd() + "/output/logistic_regression/"
np.set_printoptions(threshold=np.inf)


def main():
    util.log("Reading data")
    training = util.get_clean_dataframe_from_file(DataDir.all_tables)
    training = util.convert_input_column_type(training)

    util.log(training['attack_cat'].value_counts())

    # Binary
    # Raw training data
    x_train, y_train = util.get_input_output(training, class_type='binary')
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_logistic_regression(x_train, y_train, x_test, y_test, MODE.binary)

    # Top features from above
    x_train, y_train = util.get_input_output(training, class_type='binary')
    # x_train = x_train[[x[0] for x in top_features]]
    _, x_train = util.reduce_features(input_data=x_train,
                                      output_data=y_train,
                                      output_data_type='binary',
                                      benign_include=True)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_logistic_regression(x_train, y_train, x_test, y_test, MODE.binary_reduced)

    # PCA
    x_train, y_train = util.get_input_output(training, class_type='binary')
    feature_scale = StandardScaler()  # scale the data for use with PCA
    x_train = feature_scale.fit_transform(x_train)
    pca = PCA(0.99, random_state=42)
    x_train = pca.fit_transform(x_train)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_logistic_regression(x_train, y_train, x_test, y_test, MODE.binary_PCA)

    # Multi-class
    # Raw training data
    x_train, y_train = util.get_input_output(training, class_type='multiclass')
    util.log('Unreduced (benign included) label code mapping')
    util.log(dict(zip(y_train['attack_cat'].cat.codes, y_train['attack_cat'])))

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_logistic_regression(x_train, y_train, x_test, y_test, MODE.multi)

    # Top features from above
    x_train, y_train = util.get_input_output(training, class_type='multiclass')
    # x_train = x_train[[x[0] for x in top_features]]
    _, x_train = util.reduce_features(input_data=x_train,
                                      output_data=y_train,
                                      output_data_type='multiclass',
                                      benign_include=True)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_logistic_regression(x_train, y_train, x_test, y_test, MODE.multi_reduced)

    # PCA
    x_train, y_train = util.get_input_output(training, class_type='multiclass')
    feature_scale = StandardScaler()  # scale the data for use with PCA
    x_train = feature_scale.fit_transform(x_train)
    pca = PCA(0.99, random_state=42)
    x_train = pca.fit_transform(x_train)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_logistic_regression(x_train, y_train, x_test, y_test, MODE.multi_PCA)

    # No benign
    # Raw training data WITHOUT benign labels
    x_train, y_train = util.get_input_output(training, class_type='multiclass', benign_include=False)
    util.log('Reduced (no-benign) label code mapping')
    util.log(dict(zip(y_train['attack_cat'].cat.codes, y_train['attack_cat'])))

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_logistic_regression(x_train, y_train, x_test, y_test, MODE.no_benign)

    # Top features from above WITHOUT benign labels
    x_train, y_train = util.get_input_output(training, class_type='multiclass', benign_include=False)
    _, x_train = util.reduce_features(input_data=x_train,
                                      output_data=y_train,
                                      output_data_type='multiclass',
                                      benign_include=False)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_logistic_regression(x_train, y_train, x_test, y_test, MODE.no_benign_reduced)

    # PCA WITHOUT benign labels
    x_train, y_train = util.get_input_output(training, class_type='multiclass', benign_include=False)
    feature_scale = StandardScaler()
    x_train = feature_scale.fit_transform(x_train)
    pca = PCA(0.99, random_state=42)
    x_train = pca.fit_transform(x_train)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_logistic_regression(x_train, y_train, x_test, y_test, MODE.no_benign_PCA)


def run_logistic_regression(x_train,
                            y_train,
                            x_test,
                            y_test,
                            mode):
    clf = LogisticRegression(max_iter=1500,
                             multi_class='multinomial',
                             solver='saga',
                             penalty='l2',
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
    plt.title(f'Logistic Regression {mode.value} Confusion Matrix')
    util.log(f'Logistic Regression {mode.value} Confusion Matrix')
    util.log(display_cm.confusion_matrix)
    plt.savefig(PATH_OUTPUT + f'Logistic Regression Confusion Matrix - {mode.value}.png')
    plt.close()

    # scores
    util.log(f"Accuracy - {mode.value}: {accuracy_score(y_test, predict)}")
    # These scores only available to binary
    if MODE.is_binary(mode):
        util.log(f"F1 Score - {mode.value}: {f1_score(y_test, predict)}")
        util.log(f"Precision - {mode.value}: {precision_score(y_test, predict)}")
        util.log(f"Recall - {mode.value}: {recall_score(y_test, predict)}")

    # feature importance
    if MODE.is_raw(mode):
        util.log("Generating graph")

        # create importance dictionary as {name: importance}
        feature_dict = dict(zip(x_train.columns, np.std(x_train, 0) * clf.coef_[0]))
        feature_importance = [(k, v) for k, v in feature_dict.items()]
        sorted_list = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)

        top = sorted_list[:10]
        top_names = [x[0] for x in top]
        top_values = [x[1] for x in top]

        # output graph
        plt.barh(top_names, top_values)
        plt.gca().invert_yaxis()
        plt.title(f"Logistic regression feature importance {mode.value}")

        plt.xlabel("Importance")
        plt.ylabel("Feature name")
        for index, value in enumerate(top_values):
            plt.text(value, index, str("{:.2e}".format(float(value))))

        plt.savefig(PATH_OUTPUT + f"Logistic Regression feature importance - {mode.value}.png")
        util.log("Output: ")
        util.log(PATH_OUTPUT + f"Logistic Regression feature importance - {mode.value}.png")

        plt.close()


if __name__ == '__main__':
    main()
