import pandas as pd
import util

from constants import DataDir
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, KFold


def run_random_forest_classification(input_data: pd.DataFrame,
                                     output_data: pd.DataFrame,
                                     class_type: str) -> None:
    """
    Runs random forest classification on the given input and output data.

    :param input_data: An input as a pandas dataframe.
    :param output_data: The output labels as a pandas dataframe.
    :param class_type: The type of class. (binary, multiclass)
    :return: Nothing, a "pure" IO operation.
    """
    scoring = ['f1_weighted', 'accuracy']
    rf_clf = RandomForestClassifier(max_depth=None, n_estimators=150)
    cv = KFold(n_splits=10, random_state=42, shuffle=True)
    scores = cross_validate(rf_clf, input_data, output_data, scoring=scoring, cv=cv, n_jobs=-1)
    accuracy = scores['test_accuracy']
    f1_weighted = scores['test_f1_weighted']

    print(f'Random Forest over {class_type} classes '
          f'- Accuracy score (mean): {accuracy.mean() * 100:.5f}%, '
          f'F1 Score (Weighted) (mean): {f1_weighted.mean():.5f}')


if __name__ == '__main__':
    df = util.get_clean_dataframe_from_file(DataDir.all_tables)
    df = util.category_to_numeric(df)

    # does random forest classification on binary data
    x, y = util.get_input_output(df, class_type='binary')
    _, reduced_features = util.reduce_features(x, y, output_data_type='binary')
    run_random_forest_classification(reduced_features, y, class_type='binary')

    # does random forest classification on multiclass data
    x, y = util.get_input_output(df, class_type='multiclass')
    _, reduced_features = util.reduce_features(x, y, output_data_type='multiclass')
    run_random_forest_classification(reduced_features, y, class_type='multiclass')
