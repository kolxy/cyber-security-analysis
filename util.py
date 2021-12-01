import numpy as np
import pandas as pd
import datetime as dt
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder

from constants import DataDir
from joblib import dump, load
from os.path import exists
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from typing import Tuple


def merge_times_into_frequency(time_array: np.ndarray, time_window: int = 60) -> np.ndarray:
    """
    Merges time series data into frequencies over a given time window.

    :param time_array: A numpy array of timestamps.
    :param time_window: A time window in seconds, defaults to 60 seconds.
    :return: A new array with the new frequencies.
    """
    counts = []
    last_time = time_array[0]
    count = 1
    for index in range(1, len(time_array)):
        total_seconds = (time_array[index] - last_time) / np.timedelta64(1, 's')
        if total_seconds > time_window:
            counts.append(count)
            last_time = time_array[index]
            count = 1
        else:
            count += 1

    return np.asarray(counts)


def merge_dos_attack(time_array: np.ndarray,
                     attack_category_array: np.ndarray,
                     time_window: int = 60) -> np.ndarray:
    """
    Merge DOS attacks into a time window, aggregate time categories. If
    a DOS attack is found in that time window, then it adds that to the count of
    DoS attacks found at that time window.

    :param time_array: The time array that has the times in unmerged format.
    :param attack_category_array: The attack category array which stores the attack categories.
    :param time_window: The time window, in seconds. Defaults to 60 seconds.
    :return: The counts of the (D)DoS attacks.
    """
    counts = []
    last_time = time_array[0]
    count = 1 if attack_category_array[0] == 'DoS' else 0
    for index in range(1, len(time_array)):
        total_seconds = (time_array[index] - last_time) / np.timedelta64(1, 's')
        if total_seconds > time_window:
            counts.append(count)
            last_time = time_array[index]
            count = 0

        if attack_category_array[index] == 'DoS':
            count += 1

    return np.asarray(counts)


def get_host_connection_frequency(df: pd.DataFrame, dst_ip: str) -> pd.DataFrame:
    """
    Gets connections to a host per minute.

    :param df: A pandas dataframe containing UNSW cyber-security data.
    :param dst_ip: The destination IP that we want to query.
    :return: A new dataframe that contains the number of connections
    per minute to a given host.
    """
    new_df = df[['dstip', 'stime', 'attack_cat']]
    new_df = new_df[new_df['dstip'] == dst_ip]
    new_df = new_df.drop(columns=['dstip'])
    attack_categories = merge_dos_attack(new_df['stime'].values.ravel(), new_df['attack_cat'].values.ravel())
    frequencies = merge_times_into_frequency(new_df['stime'].values.ravel())
    times_in_seconds = np.asarray(list(range(60, (len(frequencies) + 1) * 60, 60)))
    return pd.DataFrame({'elapsed_seconds': times_in_seconds,
                         'connection_frequency': frequencies,
                         'dos_sum': attack_categories})


def get_clean_dataframe_from_file(filename: str,
                                  method='h5',
                                  key='df') -> pd.DataFrame:
    """
    Returns a clean dataframe using a specific method. Supports reading from
    CSV as well as .h5 file.

    :param filename: The name of the file.
    :param method: The method (h5|csv) for reading the file, in this case, we default
    to using .h5 for reading in our data.
    :param key: H5 ONLY - Where the data is located in the .h5 file.
    :return: The dataframe found at the filename (sorted by time)
    """
    if method == 'h5':
        df = pd.DataFrame(pd.read_hdf(filename,
                                      key=key,
                                      mode='r'))
    elif method == 'csv':
        df = pd.read_csv(filename,
                         header=None,
                         index_col=False,
                         names=DataDir.d_types.keys(),
                         converters=DataDir.d_types)
    else:
        raise ValueError(f'Invalid method: {method}. Use either csv or h5')

    df = df.dropna()
    df = df.drop_duplicates()
    df = df.sort_values('stime')

    return df


def get_mp_dataframe_from_file(filename: str,
                               method='h5',
                               key='df') -> pd.DataFrame:
    """
    Returns a clean dataframe using a specific method. Supports reading from
    CSV as well as .h5 file.

    :param filename: The name of the file.
    :param method: The method (h5|csv) for reading the file, in this case, we default
    to using .h5 for reading in our data.
    :param key: H5 ONLY - Where the data is located in the .h5 file.
    :return: The dataframe found at the filename (sorted by time)
    """
    if method == 'h5':
        df = pd.DataFrame(pd.read_hdf(filename,
                                      key=key,
                                      mode='r'))
    elif method == 'csv':
        df = pd.read_csv(filename,
                         header=None,
                         index_col=False,
                         names=DataDir.d_types.keys(),
                         converters=DataDir.d_types)
    else:
        raise ValueError(f'Invalid method: {method}. Use either csv or h5')

    df = df.sort_values('stime')

    return df


def category_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts all categorical data to from strings to numbers.

    :param df: An input dataframe with categorical data in string format.
    :return: An input dataframe with categorical data in numeric format.
    """
    # convert all strings to categorical data-types
    for c in df.columns[df.dtypes == object]:
        df[c] = df[c].astype('category')

    # convert all categorical columns to corresponding codes
    for c in df.columns[df.dtypes == 'category']:
        df[c] = df[c].cat.codes

    return df


def get_input_output_famd(df: pd.DataFrame,
                          class_type: str = 'binary') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Gets the input and the output (eit

    :param df: An input dataframe that has all numeric types.
    :param class_type: The type of classifier we are making (binary|multiclass).
    :return: The input and output data for use with sklearn models.
    """
    input_data = df.drop(['attack_cat', 'label'], axis=1)

    if class_type == 'binary':
        output_data = df['label']
    elif class_type == 'multiclass':
        output_data = df['attack_cat']
        # output_data = LabelEncoder().fit_transform(output_data)
    else:
        raise ValueError(f'Invalid class type: {class_type}. Use either binary or multiclass')
    #
    # encoder = ce.BinaryEncoder(return_df=True)
    # input_data = encoder.fit_transform(input_data)
    # input_data = input_data.astype(np.single)

    return input_data, output_data


def get_input_output_pca_regular(df: pd.DataFrame,
                                 class_type: str = 'binary') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Gets the input and the output (eit

    :param df: An input dataframe that has all numeric types.
    :param class_type: The type of classifier we are making (binary|multiclass).
    :return: The input and output data for use with sklearn models.
    """
    input_data = df.drop(['attack_cat', 'label'], axis=1)

    if class_type == 'binary':
        output_data = df['label']
    elif class_type == 'multiclass':
        output_data = df['attack_cat']
        output_data = LabelEncoder().fit_transform(output_data)
    else:
        raise ValueError(f'Invalid class type: {class_type}. Use either binary or multiclass')

    encoder = ce.BinaryEncoder(return_df=True)
    input_data = encoder.fit_transform(input_data)
    input_data = input_data.astype(np.single)

    return input_data, output_data



def reduce_features(input_data: pd.DataFrame,
                    output_data: pd.DataFrame,
                    output_data_type: str = 'binary') -> Tuple[object, pd.DataFrame]:
    """
    Reduces the features of input by performing feature selection
    with a random forest classifier.

    :param input_data: Data with full feature set.
    :param output_data: Output labels.
    :param output_data_type: The type of the output data.
    :return: Data with reduced feature set, and the random forest classifier
    for performing shapley value analysis.
    """
    if exists(f'output/{output_data_type}_rf_classifier.joblib'):
        clf = load(f'output/{output_data_type}_rf_classifier.joblib')
    else:
        clf = RandomForestClassifier(max_depth=None, n_estimators=150)
        clf = clf.fit(input_data, output_data)
        dump(clf, f'output/{output_data_type}_rf_classifier.joblib')

    return clf, SelectFromModel(clf, prefit=True).transform(input_data)


def convert_input_column_type(df: pd.DataFrame):
    """
    equivalent to pivot_wider() in R
    convert input column to all numeric types

    IMPORTANT: srcip and dstip should be dropped separately
    """

    # pivot wider for string features
    result = pd.get_dummies(df, columns=["proto", "state", "service"])

    # convert time to timestamp
    if 'stime' in result.columns:
        result['stime'] = result.stime.values.astype(int) // 10**9
        result['ltime'] = result.ltime.values.astype(int) // 10**9
    return result
