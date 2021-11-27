import pandas as pd

from constants import DataDir


def process_csv_to_h5(csv_path: str, h5_path: str) -> None:
    # read in the CSV
    df = pd.read_csv(csv_path,
                     header=None,
                     index_col=False,
                     names=DataDir.d_types.keys(),
                     converters=DataDir.d_types)
    # convert all strings to categorical data-types
    for c in df.columns[df.dtypes == object]:
        df[c] = df[c].astype('category')

    # convert to hdf5 file
    df.to_hdf(h5_path,
              key="df",
              complevel=9,
              complib='blosc',
              format="table")


def process_all_csv() -> None:
    # read in the CSV
    df = pd.concat([pd.read_csv(DataDir.table1,
                                header=None,
                                index_col=False,
                                names=DataDir.d_types.keys(),
                                converters=DataDir.d_types),
                    pd.read_csv(DataDir.table2,
                                header=None,
                                index_col=False,
                                names=DataDir.d_types.keys(),
                                converters=DataDir.d_types),
                    pd.read_csv(DataDir.table3,
                                header=None,
                                index_col=False,
                                names=DataDir.d_types.keys(),
                                converters=DataDir.d_types),
                    pd.read_csv(DataDir.table4,
                                header=None,
                                index_col=False,
                                names=DataDir.d_types.keys(),
                                converters=DataDir.d_types)])

    # convert all strings to categorical data-types
    for c in df.columns[df.dtypes == object]:
        df[c] = df[c].astype('category')

    # convert to hdf5 file
    df.to_hdf('data/UNSW-NB15_1to4.h5',
              key='df',
              complevel=9,
              complib='blosc',
              format='table')


if __name__ == '__main__':
    print('Converting all of the csv into an .h5 file.')
    process_all_csv()
