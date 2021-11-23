import pandas as pd

from constants import DataDir


def process_csv_to_h5(csv_path: str, h5_path: str) -> None:
    # read in the CSV
    df = pd.read_csv(csv_path,
                     header=None,
                     index_col=False,
                     names=DataDir.d_types.keys(),
                     converters=DataDir.d_types)

    # convert all strings to categorical datatypes
    for c in df.columns[df.dtypes == object]:
        df[c] = df[c].astype('category')

    # drop useless garbage
    df = df.dropna()
    df = df.drop_duplicates()

    # convert to hdf5 file
    df.to_hdf(h5_path,
              key="df",
              complevel=9,
              complib='blosc',
              format="table")
