from timeit import default_timer

import numpy as np
from matplotlib import pyplot as plt

import util
from constants import DataDir
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, \
    recall_score

PATH_OUTPUT = os.getcwd() + "/output/"


def main():
    print("Reading data")
    training = util.get_clean_dataframe_from_file(DataDir.all_tables)
    training = util.convert_input_column_type(training)

    print(training['attack_cat'].value_counts())
    print(dict(zip(training['attack_cat'].cat.codes, training['attack_cat'])))

    # Binary PCA

    # use training data for prediction
    x_train, y_train = util.get_input_output(training, class_type='binary')

    _, x_train_reduced = util.reduce_features(input_data=x_train, output_data=y_train, output_data_type='binary')
    print(x_train_reduced)

    print(f'Shape of x_train: {x_train.shape}, Shape of y_train {y_train.shape}')

    start_time = default_timer()

    # scale the data for use with PCA
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    # apply principal components analysis
    pca = PCA(0.99, random_state=42)
    x_train = pca.fit_transform(x_train)

    end_time = default_timer()

    print(f'Total time to run PCA (with scaling): {end_time - start_time} seconds')

    print(f'Number of PCA components: {pca.n_components_}')

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_logistic_regression(x_train, y_train, x_test, y_test, reduced=True, class_type='binary')

    # Multiclass PCA

    # use training data for prediction
    x_train, y_train = util.get_input_output(training, class_type='multiclass')

    _, x_train_reduced = util.reduce_features(input_data=x_train,
                                              output_data=y_train,
                                              output_data_type='multiclass',
                                              benign_include=True)
    print(x_train_reduced)

    print(y_train)

    # scale the data for use with PCA
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    # apply principal components analysis
    pca = PCA(0.99, random_state=42)
    x_train = pca.fit_transform(x_train)

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_logistic_regression(x_train, y_train, x_test, y_test, reduced=True, class_type='multiclass')

    # use training data for prediction
    x_train, y_train = util.get_input_output(training, class_type='multiclass', benign_include=False)

    _, x_train_reduced = util.reduce_features(input_data=x_train,
                                              output_data=y_train,
                                              output_data_type='multiclass',
                                              benign_include=False)
    print(x_train_reduced)

    # scale the data for use with PCA
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    # apply principal components analysis
    pca = PCA(0.99, random_state=42)
    x_train = pca.fit_transform(x_train)

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_logistic_regression(x_train, y_train, x_test, y_test, reduced=True, contains_benign=False, class_type='multiclass')

    # Binary No-PCA

    # use training data for prediction
    x_train, y_train = util.get_input_output(training, class_type='binary')

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_logistic_regression(x_train, y_train, x_test, y_test, reduced=False, class_type='binary')

    # Multiclass No-PCA

    # use training data for prediction
    x_train, y_train = util.get_input_output(training, class_type='multiclass')

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_logistic_regression(x_train, y_train, x_test, y_test, reduced=False, class_type='multiclass')

    # Multiclass No-PCA

    # use training data for prediction
    x_train, y_train = util.get_input_output(training, class_type='multiclass', benign_include=False)

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_logistic_regression(x_train, y_train, x_test, y_test, reduced=False, contains_benign=False, class_type='multiclass')

    # Multiclass Feature Selection

    x_train, y_train = util.get_input_output(training, class_type='multiclass', benign_include=False)
    x_train = x_train[['stcpb', 'dtcpb', 'sload']]

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

    run_logistic_regression(x_train, y_train, x_test, y_test, reduced=False, contains_benign=False, class_type='multiclass')


def run_logistic_regression(x_train,
                            y_train,
                            x_test,
                            y_test,
                            class_type='binary',
                            contains_benign=True,
                            reduction_type='PCA',
                            reduced=False):
    clf = LogisticRegression(multi_class='multinomial',
                             solver='saga',
                             max_iter=1500,
                             n_jobs=-1,
                             random_state=42)
    print(f"Fitting - reduced? {reduced} - class type? {class_type} - benign? {contains_benign}")
    start_time = default_timer()
    clf.fit(x_train, y_train)
    end_time = default_timer()
    print(f'Total time to fit: {end_time - start_time} seconds')
    print(f"Predicting - reduced? {reduced} - class type? {class_type} - benign? {contains_benign}")
    start_time = default_timer()
    predict = clf.predict(x_test)
    end_time = default_timer()
    print("Generating confusion matrix")
    print(f'Total time to predict: {end_time - start_time} seconds')
    cm = confusion_matrix(y_test, predict)
    display_cm = ConfusionMatrixDisplay(confusion_matrix=cm)
    display_cm.plot()
    if class_type == 'binary':
        if reduced:
            plt.title(f'Logistic Regression {reduction_type} {class_type} Confusion Matrix')
            plt.savefig(PATH_OUTPUT + f'random_forest_{reduction_type}_{class_type}_confusion_matrix.png')
        else:
            plt.title(f'Logistic Regression no PCA or FAMD {class_type} Confusion Matrix')
            plt.savefig(PATH_OUTPUT + f'random_forest_no_pca_or_famd_{class_type}_confusion_matrix.png')
    else:
        if contains_benign:
            if reduced:
                plt.title(f'Logistic Regression {reduction_type} {class_type} Confusion Matrix')
                plt.savefig(PATH_OUTPUT + f'random_forest_{reduction_type}_{class_type}_confusion_matrix.png')
            else:
                plt.title(f'Logistic Regression no PCA or FAMD {class_type} Confusion Matrix')
                plt.savefig(PATH_OUTPUT + f'random_forest_no_pca_or_famd_{class_type}_confusion_matrix.png')
        else:
            if reduced:
                plt.title(f'Logistic Regression {reduction_type} {class_type} Confusion Matrix - No Benign')
                plt.savefig(PATH_OUTPUT + f'random_forest_pca_{class_type}_confusion_matrix_no_benign.png')
            else:
                plt.title(f'Logistic Regression no PCA or FAMD {class_type} Confusion Matrix - No Benign')
                plt.savefig(PATH_OUTPUT + f'logistic_regression_no_pca_or_famd_{class_type}_confusion_matrix_no_benign.png')

    plt.close()
    print(f"Accuracy - {class_type}: {accuracy_score(y_test, predict)}")
    print(f"F1 Score - {class_type}: {f1_score(y_test, predict, average='micro')}")
    print(f"Precision - {class_type}: {precision_score(y_test, predict, average='micro')}")
    print(f"Recall - {class_type}: {recall_score(y_test, predict, average='micro')}")

    if not reduced:
        print("Generating graph")

        # create importance dictionary as {name: importance}
        feature_dict = dict(zip(x_train.columns, np.std(x_train, 0) * clf.coef_[0]))
        feature_importance = [(k, v) for k, v in feature_dict.items()]
        sorted_list = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)

        top = sorted_list[:10]
        top_names = [x[0] for x in top]
        top_values = [x[1] for x in top]

        # output graph
        plt.close()
        plt.barh(top_names, top_values)
        plt.gca().invert_yaxis()

        if contains_benign:
            plt.title(f"Logistic regression feature importance - {class_type}")
        else:
            plt.title(f"Logistic regression feature importance - no benign - {class_type}")

        plt.xlabel("Importance")
        plt.ylabel("Feature name")
        for index, value in enumerate(top_values):
            plt.text(value, index, str("{:.2e}".format(float(value))))

        if contains_benign:
            plt.savefig(PATH_OUTPUT + f"Logistic Regression feature importance - {class_type}.png")
            print("Output: ")
            print(PATH_OUTPUT + f"Logistic Regression feature importance - {class_type}.png")
        else:
            plt.savefig(PATH_OUTPUT + f"Logistic Regression feature importance - no benign - {class_type}.png")
            print("Output: ")
            print(PATH_OUTPUT + f"Logistic Regression feature importance - no benign - {class_type}.png")

        plt.close()

        return top


if __name__ == '__main__':
    main()
