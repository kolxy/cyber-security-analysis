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


if __name__ == "__main__":
    main()
