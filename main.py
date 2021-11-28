import numpy as np
import pandas as pd
import datetime as dt

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.callbacks import EarlyStopping

from constants import DataDir
from model_utils import create_model


def main() -> None:
    df = pd.DataFrame(pd.read_hdf(DataDir.all_tables,
                                  key='df',
                                  mode='r'))

    # convert all strings to categorical data-types
    for c in df.columns[df.dtypes == object]:
        df[c] = df[c].astype('category')

    # convert all categorical columns to corresponding codes
    for c in df.columns[df.dtypes == 'category']:
        df[c] = df[c].cat.codes

    df = df.sort_values('stime')

    df['stime'] = df['stime'].map(dt.datetime.toordinal)
    df['ltime'] = df['ltime'].map(dt.datetime.toordinal)

    df = df.dropna()

    input_data = df.drop(['attack_cat', 'label'], axis=1)
    # deal with the data type bs
    input_data = np.asarray(input_data).astype(np.float32)

    output_data = df['attack_cat']

    # perform feature selection using random forest
    clf = RandomForestClassifier(max_depth=None, n_estimators=150)
    clf = clf.fit(input_data, output_data)
    model = SelectFromModel(clf, prefit=True)
    input_data_reduced = model.transform(input_data)

    # using SAGA because there is a large dataset
    scoring = ['f1_weighted', 'accuracy']
    log_reg = LogisticRegression(max_iter=500, solver='saga')
    scores = cross_validate(log_reg, input_data_reduced, output_data, scoring=scoring)
    accuracy = scores['test_accuracy']
    f1_weighted = scores['test_f1_weighted']

    print(f'Logistic Regression - Accuracy score (mean): {accuracy.mean() * 100:.5f}%, '
          f'F1 Score (Weighted) (mean): {f1_weighted.mean():.5f}')

    r_clf = RandomForestClassifier(max_depth=None, n_estimators=150)
    scores = cross_validate(r_clf, input_data_reduced, output_data, scoring=scoring)
    accuracy = scores['test_accuracy']
    f1_weighted = scores['test_f1_weighted']

    print(f'Random Forest - Accuracy score (mean): {accuracy.mean() * 100:.5f}%, '
          f'F1 Score (Weighted) (mean): {f1_weighted.mean():.5f}')

    ada_clf = AdaBoostClassifier(n_estimators=150)
    scores = cross_validate(ada_clf, input_data_reduced, output_data, scoring=scoring)
    accuracy = scores['test_accuracy']
    f1_weighted = scores['test_f1_weighted']

    print(f'AdaBoost - Accuracy score (mean): {accuracy.mean() * 100:.5f}%, '
          f'F1 Score (Weighted) (mean): {f1_weighted.mean():.5f}')

    # one hot encode the stuff
    encoder = LabelEncoder()
    encoder.fit(output_data)
    output_data = output_data.transform(output_data)
    output_data = np_utils.to_categorical(output_data, dtype=np.int8)

    # cv instead (for everything but MLP)?
    x_train, x_test, y_train, y_test = train_test_split(input_data_reduced, output_data, test_size=0.25)

    model = create_model(x_train.shape[1], y_train.shape[1])
    callback = EarlyStopping(monitor='loss', patience=5)
    model.fit(x_train,
              y_train,
              epochs=200,
              batch_size=500,
              callbacks=[callback], verbose=0)
    score = model.evaluate(x_test, y_test, batch_size=500)
    print(f'Accuracy score (Multilayer Perceptron): {score[1] * 100:.5f}%, test loss: {score[0]:.5f}')

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
