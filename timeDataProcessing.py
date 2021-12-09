import numpy as np
import pandas as pd
from sklearn import preprocessing as skpp
from enum import Enum

import util
from constants import DataDir
import constants as gv

def window_stack(arr, width=16, stepsize=None):
    if stepsize is None:
        stepsize = int(width/2)

    nWindows = int(arr.shape[0] / stepsize -1)
    indexer = np.arange(width)[None, :] + stepsize*np.arange(nWindows)[:,None]
    arr = arr[indexer]
    return arr

def get_df():
    print("Reading data")
    df = util.get_clean_dataframe_from_file(DataDir.all_tables)
    df = util.convert_input_column_type(df)
    return df

def get_min_max_data(df:pd.DataFrame = None):
    if df is None:
        df = get_df()

    data = gv.x_y(*util.get_input_output(df, class_type='binary'))
    scaler = skpp.MinMaxScaler()
    data.x = scaler.fit_transform(data.x)
    return data


class network_window:
    class window_type(Enum):
        ALL_BENIGN = 0
        ALL_HETERO = 1
        MIXED = 2

    def __init__(self, packetWindows, nMaliciousGivenWindow):
        self.windows = packetWindows
        self.nMalicious = nMaliciousGivenWindow
        self.windowType = self.get_window_type()

    def get_window_type(self):
        if self.nMalicious.sum(axis=-1) == 0:
            return network_window.window_type.ALL_BENIGN
        elif 0 in self.nMalicious:
            return network_window.window_type.MIXED
        else:
            return network_window.window_type.ALL_HETERO

    def get_at_least_n_malicious(self, nMalicious):
        assert self.windows.shape[0] == self.nMalicious.shape[0]
        return self.windows[self.nMalicious >= nMalicious]

    def get_n_malicious(self, nMalicious):
        assert self.windows.shape[0] == self.nMalicious.shape[0]
        return self.windows[self.nMalicious == nMalicious]

    def _get_homogeneous_benign_mask(self):
        assert self.windows.shape[0] == self.nMalicious.shape[0]
        return self.nMalicious == 0

    def get_homogeneous_benign(self):
        homoMask = self._get_homogeneous_benign_mask()
        nw = network_window(
            self.windows[homoMask], self.nMalicious[homoMask]
        )
        assert nw.windowType == network_window.window_type.ALL_BENIGN
        return nw

    def get_only_heterogeneous(self):
        heteroMask = ~self._get_homogeneous_benign_mask()
        nw = network_window(
            self.windows[heteroMask], self.nMalicious[heteroMask]
        )
        assert nw.windowType == network_window.window_type.ALL_HETERO
        return nw

    @staticmethod
    def get_window_data(nTimeSteps, firstN=None):
        data = get_min_max_data()
        data.x = window_stack(data.x[:firstN], nTimeSteps)
        data.y = window_stack(data.y.to_numpy()[:firstN], nTimeSteps)
        data.y = data.y.sum(axis=-1)
        nw = network_window(data.x, data.y)
        return nw

