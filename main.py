import numpy as np
import pandas as pd
import datetime as dt
import tensorflow as tf

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.utils import np_utils

from constants import DataDir
from util import create_model


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

    input_data = df.drop(['attack_cat', 'label', 'stime', 'ltime'], axis=1)
    # deal with the data type bs
    input_data = np.asarray(input_data).astype(np.float32)

    output_data = df['attack_cat']
    x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.25)

    log_reg = LogisticRegression(max_iter=500, solver='saga')
    log_reg.fit(x_train, y_train)
    score = log_reg.score(x_test, y_test)
    print(f'Accuracy score (Logistic Regression): {score:.5f}')

    r_clf = RandomForestClassifier(max_depth=None, n_estimators=150)
    r_clf.fit(x_train, y_train)
    score = r_clf.score(x_test, y_test)
    print(f'Accuracy score (Random Forest): {score:.5f}')

    ada_clf = AdaBoostClassifier(n_estimators=150)
    ada_clf.fit(x_train, y_train)
    score = ada_clf.score(x_test, y_test)
    print(f'Accuracy score (AdaBoost): {score:.5f}')

    encoder = LabelEncoder()
    encoder.fit(output_data)
    y_train = encoder.transform(y_train)
    y_train = np_utils.to_categorical(y_train, dtype=np.int8)
    y_test = encoder.transform(y_test)
    y_test = np_utils.to_categorical(y_test, dtype=np.int8)

    model = create_model(x_train.shape[1], y_train.shape[1])
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model.fit(x_train, y_train, epochs=200, batch_size=500, callbacks=[callback])
    test_loss, test_accuracy = model.evaluate((x_test, y_test))
    print(f'Accuracy score (Multilayer Perceptron): {test_accuracy * 100}, test loss: {test_loss}')

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
