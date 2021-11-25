from typing import Union

import numpy as np
import pandas as pd
from pandas.core.arrays import ExtensionArray


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
                     attack_category_array,
                     time_window: int = 60) -> np.ndarray:
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


def get_unique_hosts(df: pd.DataFrame) -> Union[np.ndarray, ExtensionArray]:
    return df['dstip'].unique()


def convert_input_column_type(df: pd.DataFrame):
    """
    convert input column to all numeric types

    IMPORTANT: srcip and dstip should be dropped separately
    """

    # pivot wider for string features
    result = pd.get_dummies(df, columns=["proto", "state", "service"])

    # convert time to timestamp
    result['stime'] = result.stime.values.astype(int) // 10**9
    result['ltime'] = result.ltime.values.astype(int) // 10**9
    return result