import pandas as pd
import util

from constants import DataDir
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, KFold


def run_logistic_regression(input_data: pd.DataFrame,
                            output_data: pd.DataFrame,
                            class_type: str) -> None:
    """
    Runs logistic regression on the given input and output data.

    :param input_data: An input as a pandas dataframe.
    :param output_data: The output labels as a pandas dataframe.
    :param class_type: The type of class. (binary, multiclass)
    :return: Nothing, a "pure" IO operation.
    """
    # using SAGA because there is a large dataset
    scoring = ['f1_weighted', 'accuracy']
    log_reg = LogisticRegression(max_iter=500, solver='saga')
    cv = KFold(n_splits=5, random_state=42, shuffle=True)
    scores = cross_validate(log_reg, input_data, output_data, scoring=scoring, cv=cv, n_jobs=-1)
    accuracy = scores['test_accuracy']
    f1_weighted = scores['test_f1_weighted']

    print(f'Logistic Regression over {class_type} classes '
          f'- Accuracy score (mean): {accuracy.mean() * 100:.5f}%, '
          f'F1 Score (Weighted) (mean): {f1_weighted.mean():.5f}')


if __name__ == '__main__':
    df = util.get_clean_dataframe_from_file(DataDir.all_tables)
    df = util.category_to_numeric(df)

    # does logistic regression on binary data
    x, y = util.get_input_output(df, class_type='binary')
    _, reduced_features = util.reduce_features(x, y, output_data_type='binary')
    run_logistic_regression(reduced_features, y, class_type='binary')

    # does logistic regression on multiclass data
    x, y = util.get_input_output(df, class_type='multiclass')
    _, reduced_features = util.reduce_features(x, y, output_data_type='multiclass')
    run_logistic_regression(reduced_features, y, class_type='multiclass')
