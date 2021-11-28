import pandas as pd
from sklearn.inspection import permutation_importance

import util
from constants import DataDir


def compute_feature_importance(input_data: pd.DataFrame,
                               output_data: pd.DataFrame,
                               random_forest_classifier: object,
                               class_type: str = 'binary'):
    r = permutation_importance(random_forest_classifier,
                               input_data,
                               output_data,
                               n_repeats=10,
                               random_state=42,
                               n_jobs=-1)
    print(f'{class_type} class feature importance:')

    # using an sklearn example of displaying feature importance using the permutation method
    # https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-importance
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f'{df.columns[i]:<8}'
                  f'{r.importances_mean[i]:.3f}'
                  f'+/- {r.importances_std[i]:.3f}')


if __name__ == '__main__':
    df = util.get_clean_dataframe_from_file(DataDir.all_tables)
    df = util.category_to_numeric(df)

    # computes binary class feature importance (label column)
    x, y = util.get_input_output(df, class_type='binary')
    rf_clf, _ = util.reduce_features(x, y, output_data_type='binary')
    compute_feature_importance(x, y, rf_clf, class_type='binary')

    # computes multiclass feature importance (attack_cat column)
    x, y = util.get_input_output(df, class_type='multiclass')
    rf_clf, _ = util.reduce_features(x, y, output_data_type='multiclass')
    compute_feature_importance(x, y, rf_clf, class_type='multiclass')
