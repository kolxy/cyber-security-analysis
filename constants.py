import datetime
import numpy as np
import sys

from dataclasses import dataclass


# handles the integer edge cases
def parse_integers(x: str) -> int:
    try:
        return np.int(x)
    except ValueError:
        return np.int(x, 16) if '0x' in x else np.NaN


# handles the attack category parsing
def parse_attack_cat(x: str) -> str:
    if x == '':
        return 'benign'

    # a weird case in their labeling
    if x.lower().strip() == 'backdoor':
        return 'backdoors'

    return x.lower().strip()


# data file directory (stored as an immutable dataclass)
@dataclass(frozen=True)
class DataDir:
    table1 = "data/UNSW-NB15_1.csv"
    table1h5 = "data/unsw1.h5"
    table2 = "data/UNSW-NB15_2.csv"
    table3 = "data/UNSW-NB15_3.csv"
    table4 = "data/UNSW-NB15_4.csv"
    all_tables = "data/UNSW-NB15_1to4.h5"
    features = "data/NUSW-NB15_features.csv"
    gt = "data/NUSW-NB15_GT.csv"
    events = "data/UNSW-NB_LIST_EVENTS.csv"
    sample_train = "data/sample/UNSW_NB15_training-set.csv"
    sample_test = "data/sample/UNSW_NB15_testing-set.csv"

    # data types that are used in each column
    d_types = {
        'srcip': str,
        'sport': str,
        'dstip': str,
        'dsport': str,
        'proto': str,
        'state': str,
        'dur': np.single,
        'sbytes': parse_integers,
        'dbytes': parse_integers,
        'sttl': parse_integers,
        'dttl': parse_integers,
        'sloss': parse_integers,
        'dloss': parse_integers,
        'service': str,
        'sload': np.single,
        'dload': np.single,
        'spkts': parse_integers,
        'dpkts': parse_integers,
        'swin': parse_integers,
        'dwin': parse_integers,
        'stcpb': parse_integers,
        'dtcpb': parse_integers,
        'smeansz': parse_integers,
        'dmeansz': parse_integers,
        'trans_depth': parse_integers,
        'res_bdy_len': parse_integers,
        'sjit': np.single,
        'djit': np.single,
        'stime': lambda x: datetime.datetime.fromtimestamp(int(x)),
        'ltime': lambda x: datetime.datetime.fromtimestamp(int(x)),
        'sintpkt': np.single,
        'dintpkt': np.single,
        'tcprtt': np.single,
        'synack': np.single,
        'ackdat': np.single,
        'is_sm_ips_ports': bool,
        'ct_state_ttl': parse_integers,
        'ct_flw_http_mthd': parse_integers,
        'is_ftp_login': bool,
        'ct_ftp_cmd': parse_integers,
        'ct_srv_src': parse_integers,
        'ct_srv_dst': parse_integers,
        'ct_dst_ltm': parse_integers,
        'ct_src_ltm': parse_integers,
        'ct_src_dport_ltm': parse_integers,
        'ct_dst_sport_ltm': parse_integers,
        'ct_dst_src_ltm': parse_integers,
        'attack_cat': parse_attack_cat,
        'label': parse_integers
    }

def enable_tf_debug(eager: object = True, debugMode: object = True) -> object:
    import tensorflow as tf
    tf.config.run_functions_eagerly(eager)
    if debugMode: tf.data.experimental.enable_debug_mode()

def tf_np_behavior():
    import tensorflow.python.ops.numpy_ops.np_config as np_config
    np_config.enable_numpy_behavior()

class x_y:
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

    def transform(self, function, *args, **kwargs):
        if self.x is not None:
            self.x = function(self.x, *args, **kwargs)
        if self.y is not None:
            self.y = function(self.y, *args, **kwargs)

class ml_data:
    def __init__(self, train=None, test=None, validate=None):
        self.train = train
        self.test = test
        self.validate = validate

    def apply(self, function, *args, **kwargs):
        returns = ml_data()
        if self.train is not None:
            returns.train = function(self.train, *args, **kwargs)
        if self.test is not None:
            returns.test = function(self.test, *args, **kwargs)
        if self.validate is not None:
            returns.validate = function(self.validate, *args, **kwargs)
        return returns

    def transform(self, function, *args, **kwargs):
        if self.train is not None:
            self.train = function(self.train, *args, **kwargs)
        if self.test is not None:
            self.test = function(self.test, *args, **kwargs)
        if self.validate is not None:
            self.validate = function(self.validate, *args, **kwargs)

def _get_debug_flag():
    return sys.gettrace() is not None

DEBUG = _get_debug_flag()

#if cuda is working with tensorflow, this sets gpu0 to NOT be visible
def disable_gpu():
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

#if cuda is working with tensorflow, this sets gpu0 to be visible
def enable_gpu():
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"