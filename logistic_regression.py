import pandas as pd
import util

from constants import DataDir
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate


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
    scores = cross_validate(log_reg, input_data, output_data, scoring=scoring)
    accuracy = scores['test_accuracy']
    f1_weighted = scores['test_f1_weighted']

    print(f'Logistic Regression over {class_type} classes '
          f'- Accuracy score (mean): {accuracy.mean() * 100:.5f}%, '
          f'F1 Score (Weighted) (mean): {f1_weighted.mean():.5f}')


if __name__ == '__main__':
    df = util.get_clean_dataframe_from_file(DataDir.all_tables)
    df = util.category_to_numeric(df)
    x, y = util.get_input_output(df, class_type='binary')
    rf_clf, reduced_features = util.reduce_features(x, y, output_data_type='binary')
    r = permutation_importance(rf_clf,
                               x,
                               y,
                               n_repeats=30,
                               random_state=42)
    print('Binary class feature importances: ')

    # using an sklearn example of displaying feature importance using the permutation method
    # https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-importance
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f'{df.columns[i]:<8}'
                  f'{r.importances_mean[i]:.3f}'
                  f'+/- {r.importances_std[i]:.3f}')

    run_logistic_regression(reduced_features, y, class_type='binary')
    x, y = util.get_input_output(df, class_type='multiclass')
    rf_clf, reduced_features = util.reduce_features(x, y, output_data_type='multiclass')
    r = permutation_importance(rf_clf,
                               x,
                               y,
                               n_repeats=30,
                               random_state=42)
    print('Multiclass feature importances: ')

    # using an sklearn example of displaying feature importance using the permutation method
    # https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-importance
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f'{df.columns[i]:<8}'
                  f'{r.importances_mean[i]:.3f}'
                  f'+/- {r.importances_std[i]:.3f}')

    run_logistic_regression(reduced_features, y, class_type='multiclass')
