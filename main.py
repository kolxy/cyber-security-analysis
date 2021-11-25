import numpy as np
import pandas as pd
import datetime as dt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from constants import DataDir


def main() -> None:
    df = pd.DataFrame(pd.read_hdf(DataDir.all_tables,
                      key='df',
                      mode='r'))
    # convert all strings to categorical datatypes
    for c in df.columns[df.dtypes == object]:
        df[c] = df[c].astype('category')

    for c in df.columns[df.dtypes == 'category']:
        df[c] = df[c].cat.codes

    df['stime'] = df['stime'].map(dt.datetime.toordinal)
    df['ltime'] = df['ltime'].map(dt.datetime.toordinal)

    df = df.dropna()

    input_data = df.drop(['attack_cat', 'label'], axis=1)
    output_data = df['label'].astype(np.int8).to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.25)

    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)
    score = log_reg.score(x_test, y_test)
    print(f'Accuracy score: {score:.2f}')

    # matrix profile shit
    # hosts = get_unique_hosts(df)
    # connection_frequencies = get_host_connection_frequency(df, hosts[0])
    # profile_by_minutes = 25
    # print(df['attack_cat'].head)
    # print(df.dtypes)
    # profile = mp.compute(connection_frequencies['connection_frequency'].values, profile_by_minutes, n_jobs=-1)
    # profile = mp.discover.discords(profile)
    #
    # # We have to adjust the matrix profile to match the dimensions of the original
    # # time series
    # mp_adjusted = np.append(profile['mp'], np.zeros(profile['w'] - 1) + np.nan)
    #
    # # Create a plot with three subplots
    # fig, ax = plt.subplots(1, 1, figsize=(20, 7))
    # ax.plot(np.arange(len(profile['data']['ts'])), profile['data']['ts'])
    # ax.set_title('Connection Frequencies', size=22)
    # ax.set_xlabel('Time elapsed (minutes)')
    # ax.set_ylabel('Number of connections')
    #
    # for discord in profile['discords']:
    #     print(connection_frequencies['dos_sum'][discord])
    #     x = np.arange(discord, discord + profile['w'])
    #     y = profile['data']['ts'][discord:discord + profile['w']]
    #
    #     ax.plot(x, y, c='r')
    #
    # plt.show()


if __name__ == "__main__":
    main()
