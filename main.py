import numpy as np
import stumpy
from matplotlib import pyplot as plt

import pandas as pd
from constants import DataDir


def main():
    df = pd.DataFrame(pd.read_hdf(DataDir.all_tables,
                      key='df',
                      mode='r'))

    # do general cleansing
    df = df.dropna()
    df = df.drop_duplicates()
    print(df.shape)
    print(df['dbytes'].head)
    dest_bytes = df['dbytes'].astype(np.float64)
    m = 32
    mp = stumpy.gpu_stump(dest_bytes, m)
    print(mp)
    plt.show()


if __name__ == "__main__":
    main()
