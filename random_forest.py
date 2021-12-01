import category_encoders as ce
import numpy as np
import util

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from constants import DataDir
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def run_random_forest_classification(x_train: np.ndarray,
                                     y_train: np.ndarray,
                                     x_test: np.ndarray,
                                     y_test: np.ndarray,
                                     class_type: str) -> None:
    rf_clf = RandomForestClassifier(max_depth=None,
                                    n_estimators=150,
                                    n_jobs=-1)
    print("Fitting")
    rf_clf.fit(x_train, y_train)
    print("Predicting")
    score = rf_clf.score(x_test, y_test)
    print(f"Accuracy - {class_type}: {score}")


if __name__ == '__main__':
    print("Reading data")
    training = util.get_clean_dataframe_from_file(DataDir.all_tables)
    training = util.convert_input_column_type(training)

    # use training data for prediction
    x_train, y_train = util.get_input_output(training, class_type='multiclass')

    # fix types
    # binary encode string features
    encoder = ce.BinaryEncoder(return_df=True)
    x_train = encoder.fit_transform(x_train)
    y_train = LabelEncoder().fit_transform(y_train)

    # scale the data for use with PCA
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    # apply principal components analysis
    pca = PCA(0.99)
    x_train = pca.fit_transform(x_train)

    print(pca.n_components_)

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25)

    run_random_forest_classification(x_train, y_train, x_test, y_test, class_type='multiclass')

