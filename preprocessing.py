import pandas as pd
import os

from constants import DataDir

# read in the CSV
df = pd.read_csv(DataDir.table1,
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
df.to_hdf(os.path.join('data', 'unsw1.h5'),
          key="df",
          complevel=9,
          complib='blosc',
          format="table")
