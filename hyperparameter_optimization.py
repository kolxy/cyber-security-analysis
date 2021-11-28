import pandas as pd
from scipy.stats import loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, KFold
from joblib import dump

import util
from constants import DataDir


def optimize_logistic_regression(input_data: pd.DataFrame,
                                 output_data: pd.DataFrame,
                                 class_type: str = 'binary'):
    """
    Optimizes logistic regression hyper-parameters. Code adapted from
    https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/

    :param input_data: The input data.
    :param output_data: The output data.
    :param class_type: The type of the class (binary|multiclass).
    :return: Nothing, a "pure" IO operation.
    """
    model = LogisticRegression()
    optimization_parameters = dict()
    optimization_parameters['solver'] = ['newton-cg', 'sag', 'saga', 'lbfgs', 'liblinear']
    optimization_parameters['penalty'] = ['l1', 'l2', 'elasticnet']
    optimization_parameters['reg_strength_inverse'] = loguniform(1e-5, 1000)
    cv = KFold(n_splits=10, random_state=42)
    randomized_search = RandomizedSearchCV(model,
                                           optimization_parameters,
                                           n_iter=100,
                                           scoring='accuracy',
                                           n_jobs=-1,
                                           cv=cv,
                                           random_state=42)
    result = randomized_search.fit(input_data, output_data)
    dump(result, f'output/{class_type}_logistic_regression_hpoptimized.joblib')
    print(f'Best score: {result.best_score_}')
    print(f'Best hyper-parameters for logistic regression: {result.best_params_}')


def optimize_random_forest():
    pass


def optimize_multilayer_perceptron():
    pass


if __name__ == 'main':
    df = util.get_clean_dataframe_from_file(DataDir.all_tables)
    df = util.category_to_numeric(df)

    # optimize logistic regression hyper-parameters - binary
    x, y = util.get_input_output(df, class_type='binary')
    _, reduced_features = util.reduce_features(x, y, output_data_type='binary')
    optimize_logistic_regression(reduced_features, y, class_type='binary')

    # optimize logistic regression hyper-parameters - multiclass
    x, y = util.get_input_output(df, class_type='multiclass')
    _, reduced_features = util.reduce_features(x, y, output_data_type='multiclass')
    optimize_logistic_regression(reduced_features, y, class_type='multiclass')
