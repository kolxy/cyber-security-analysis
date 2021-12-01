import numpy as np
from matplotlib import pyplot as plt

import util
from constants import DataDir
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

PATH_OUTPUT = os.getcwd() + "/output/"


def main():
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
    run_logistic_regression(x_train, y_train, x_test, y_test, reduced=True, class_type='binary')

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
    run_logistic_regression(x_train, y_train, x_test, y_test, reduced=True, class_type='multiclass')

    # Binary No-PCA

    # use training data for prediction
    x_train, y_train = util.get_input_output(training, class_type='binary')

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25)
    run_logistic_regression(x_train, y_train, x_test, y_test, reduced=False, class_type='binary')

    # Multiclass No-PCA

    # use training data for prediction
    x_train, y_train = util.get_input_output(training, class_type='multiclass')

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25)
    run_logistic_regression(x_train, y_train, x_test, y_test, reduced=False, class_type='multiclass')


def run_logistic_regression(x_train,
                            y_train,
                            x_test,
                            y_test,
                            class_type='binary',
                            reduced=False):
    clf = LogisticRegression(multi_class='multinomial',
                             solver='saga',
                             max_iter=1000,
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
        plt.savefig(PATH_OUTPUT + f'logistic_regression_pca_{class_type}_confusion_matrix.png')
    else:
        plt.savefig(PATH_OUTPUT + f'logistic_regression_no_pca_{class_type}_confusion_matrix.png')

    print(f"Accuracy - {class_type}: {accuracy_score(y_test, predict)}")

    if not reduced:
        print("Generating graph")

        # create importance dictionary as {name: importance}
        feature_dict = dict(zip(x_train.columns, np.std(x_train, 0) * clf.coef_[0]))
        feature_importance = [(k, v) for k, v in feature_dict.items()]
        sorted_list = sorted(feature_importance, key = lambda x: abs(x[1]), reverse = True)

        top = sorted_list[:10]
        top_names = [x[0] for x in top]
        top_values = [x[1] for x in top]

        # output graph
        plt.barh(top_names, top_values)
        plt.gca().invert_yaxis()
        plt.title("Logistic regression feature importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature name")
        for index, value in enumerate(top_values):
            plt.text(value, index, str("{:.2e}".format(float(value))))
        plt.savefig(PATH_OUTPUT + "Logistic Regression feature importance.png")
        print("Output: ")
        print(PATH_OUTPUT + "Logistic Regression feature importance.png")

        return top


if __name__ == '__main__':
    main()
