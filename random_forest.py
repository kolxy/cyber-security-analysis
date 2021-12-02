import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, \
    recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import util
from constants import DataDir

PATH_OUTPUT = os.getcwd() + "/output/"


def run_random_forest_classification(x_train: np.ndarray,
                                     y_train: np.ndarray,
                                     x_test: np.ndarray,
                                     y_test: np.ndarray,
                                     class_type: str,
                                     contains_benign: bool,
                                     reduced: bool) -> None:
    clf = RandomForestClassifier(max_depth=None,
                                 n_estimators=150,
                                 random_state=42,
                                 n_jobs=-1)
    print(f"Fitting - reduced? {reduced} - class type? {class_type} - benign? {contains_benign}")
    clf.fit(x_train, y_train)
    print(f"Predicting - reduced? {reduced} - class type? {class_type} - benign? {contains_benign}")
    predict = clf.predict(x_test)
    print("Generating confusion matrix")
    cm = confusion_matrix(y_test, predict)
    display_cm = ConfusionMatrixDisplay(confusion_matrix=cm)
    display_cm.plot()
    if class_type == 'binary':
        if reduced:
            plt.title(f'Random Forest PCA {class_type} Confusion Matrix')
            plt.savefig(PATH_OUTPUT + f'random_forest_pca_{class_type}_confusion_matrix.png')
        else:
            plt.title(f'Random Forest no PCA {class_type} Confusion Matrix')
            plt.savefig(PATH_OUTPUT + f'random_forest_no_pca_{class_type}_confusion_matrix.png')
    else:
        if contains_benign:
            if reduced:
                plt.title(f'Random Forest PCA {class_type} Confusion Matrix')
                plt.savefig(PATH_OUTPUT + f'random_forest_pca_{class_type}_confusion_matrix.png')
            else:
                plt.title(f'Random Forest no PCA {class_type} Confusion Matrix')
                plt.savefig(PATH_OUTPUT + f'random_forest_no_pca_{class_type}_confusion_matrix.png')
        else:
            if reduced:
                plt.title(f'Random Forest PCA {class_type} Confusion Matrix - No Benign')
                plt.savefig(PATH_OUTPUT + f'random_forest_pca_{class_type}_confusion_matrix_no_benign.png')
            else:
                plt.title(f'Random Forest no PCA {class_type} Confusion Matrix - No Benign')
                plt.savefig(PATH_OUTPUT + f'random_forest_no_pca_{class_type}_confusion_matrix_no_benign.png')

    plt.close()
    print(f"Accuracy - {class_type}: {accuracy_score(y_test, predict)}")
    print(f"F1 Score - {class_type}: {f1_score(y_test, predict, average='micro')}")
    print(f"Precision - {class_type}: {precision_score(y_test, predict, average='micro')}")
    print(f"Recall - {class_type}: {recall_score(y_test, predict, average='micro')}")


if __name__ == '__main__':
    print("Reading data")
    training = util.get_clean_dataframe_from_file(DataDir.all_tables)
    training = util.convert_input_column_type(training)

    print(training['attack_cat'].value_counts())
    print(training['attack_cat'].cat.codes)
    print(training['attack_cat'].cat.categories)
    print(dict(zip(training['attack_cat'].cat.codes, training['attack_cat'])))

    # Binary PCA

    # use training data for prediction
    x_train, y_train = util.get_input_output(training, class_type='binary')

    # scale the data for use with PCA
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    # apply principal components analysis
    pca = PCA(0.99, random_state=42)
    x_train = pca.fit_transform(x_train)

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_random_forest_classification(x_train, y_train, x_test, y_test, reduced=True, contains_benign=True,
                                     class_type='binary')

    # Multiclass PCA

    # use training data for prediction
    x_train, y_train = util.get_input_output(training, class_type='multiclass')

    # scale the data for use with PCA
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    # apply principal components analysis
    pca = PCA(0.99, random_state=42)
    x_train = pca.fit_transform(x_train)

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_random_forest_classification(x_train, y_train, x_test, y_test, reduced=True, contains_benign=True,
                                     class_type='multiclass')

    # use training data for prediction
    x_train, y_train = util.get_input_output(training, class_type='multiclass', benign_include=False)

    # scale the data for use with PCA
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    # apply principal components analysis
    pca = PCA(0.99, random_state=42)
    x_train = pca.fit_transform(x_train)

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_random_forest_classification(x_train, y_train, x_test, y_test, reduced=True, contains_benign=False,
                                     class_type='multiclass')

    # Binary No-PCA

    # use training data for prediction
    x_train, y_train = util.get_input_output(training, class_type='binary')

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_random_forest_classification(x_train, y_train, x_test, y_test, reduced=False, contains_benign=True,
                                     class_type='binary')

    # Multiclass No-PCA

    # use training data for prediction
    x_train, y_train = util.get_input_output(training, class_type='multiclass')

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_random_forest_classification(x_train, y_train, x_test, y_test, reduced=False, contains_benign=True,
                                     class_type='multiclass')

    # Multiclass No-PCA

    # use training data for prediction
    x_train, y_train = util.get_input_output(training, class_type='multiclass', benign_include=False)

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    run_random_forest_classification(x_train, y_train, x_test, y_test, reduced=True, contains_benign=True,
                                     class_type='multiclass')
