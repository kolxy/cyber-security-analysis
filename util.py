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


def get_host_connection_frequency(df: pd.DataFrame, dst_ip: str) -> pd.DataFrame:
    """
    Gets connections to a host per minute.

    :param df: A pandas dataframe containing UNSW cyber-security data.
    :param dst_ip: The destination IP that we want to query.
    :return: A new dataframe that contains the number of connections
    per minute to a given host.
    """
    new_df = df[['dstip', 'stime']]
    new_df = new_df[new_df['dstip'] == dst_ip]
    new_df = new_df.drop(columns=['dstip'])
    frequencies = merge_times_into_frequency(new_df.values.ravel())
    times_in_seconds = np.asarray(list(map(lambda x: x * 60, range(1, len(frequencies) + 1))))
    return pd.DataFrame({'elapsed_seconds': times_in_seconds, 'connection_frequency': frequencies})


def get_unique_hosts(df: pd.DataFrame) -> Union[np.ndarray, ExtensionArray]:
    return df['dstip'].unique()
