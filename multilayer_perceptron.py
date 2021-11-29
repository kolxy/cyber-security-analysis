import pandas as pd
from sklearn.neural_network import MLPClassifier

import util

from constants import DataDir
from sklearn.model_selection import cross_validate, KFold


def run_mlp_classification(input_data: pd.DataFrame,
                           output_data: pd.DataFrame,
                           class_type: str) -> None:
    """
    Runs a multilayer perceptron on the given input and output data.

    :param input_data: An input as a pandas dataframe.
    :param output_data: The output labels as a pandas dataframe.
    :param class_type: The type of class. (binary, multiclass)
    :return: Nothing, a "pure" IO operation.
    """
    scoring = ['f1_weighted', 'accuracy']
    mlp_clf = MLPClassifier(hidden_layer_sizes=(50, 50),
                            activation='relu',
                            solver='adam',
                            learning_rate_init=1e-3,
                            early_stopping=True)
    cv = KFold(n_splits=5, random_state=42, shuffle=True)
    scores = cross_validate(mlp_clf, input_data, output_data, scoring=scoring, cv=cv, n_jobs=-1)
    accuracy = scores['test_accuracy']
    f1_weighted = scores['test_f1_weighted']

    print(f'Multilayer perceptron over {class_type} classes '
          f'- Accuracy score (mean): {accuracy.mean() * 100:.5f}%, '
          f'F1 Score (Weighted) (mean): {f1_weighted.mean():.5f}')


if __name__ == '__main__':
    df = util.get_clean_dataframe_from_file(DataDir.all_tables)
    df = util.category_to_numeric(df)

    # does LinearSVC classification on binary data
    x, y = util.get_input_output(df, class_type='binary')
    _, reduced_features = util.reduce_features(x, y, output_data_type='binary')
    run_mlp_classification(reduced_features, y, class_type='binary')

    # does LinearSVC classification on multiclass data
    x, y = util.get_input_output(df, class_type='multiclass')
    _, reduced_features = util.reduce_features(x, y, output_data_type='multiclass')
    run_mlp_classification(reduced_features, y, class_type='multiclass')
