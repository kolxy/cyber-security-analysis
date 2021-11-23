import datetime
import numpy as np

from dataclasses import dataclass


# handles the integer edge cases
def parse_integers(x: str) -> int:
    try:
        return np.int(x)
    except ValueError:
        return np.int(x, 16) if '0x' in x else np.NaN


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
        'sport': parse_integers,
        'dstip': str,
        'dsport': parse_integers,
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
        'ct_src_src': parse_integers,
        'ct_src_dst': parse_integers,
        'ct_dst_ltm': parse_integers,
        'ct_src_ltm': parse_integers,
        'ct_src_dport_ltm': parse_integers,
        'ct_dst_sport_ltm': parse_integers,
        'attack_cat': str,
        'label': np.bool
    }
