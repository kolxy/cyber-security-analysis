#80/20 Hillary graph plots

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as l
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing as skpp
import numpy as np

import util
from constants import DataDir
import constants as gv

NTIMESTEPS = 16
_DROPOUT_DEFAULT = 0.25
_LEAKY_ALPHA_DEFAULT = 0.2
_OPTIMIZER = keras.optimizers.Adam(2e-4, 0.5)
_LOSSFUNC = keras.losses.MeanSquaredError()
_KERNELINIT = keras.initializers.RandomNormal(stddev=0.02)
_METRICS = [keras.metrics.RootMeanSquaredError(),
            keras.losses.MeanAbsolutePercentageError(),
            keras.losses.CosineSimilarity(),]

class min_max():
    def __init__(self, xy:gv.x_y):
        self.scaler = StandardScaler()
        self.xy = xy
        self.xy.x = self.scaler.fit_transform(self.xy.x)


def get_data():
    print("Reading data")
    df = util.get_clean_dataframe_from_file(DataDir.all_tables)
    df = util.convert_input_column_type(df)

    data = gv.x_y(*util.get_input_output(df, class_type='binary'))
    data = min_max(data)
    return data


def upsample_block(
    x,
    filters,
    activation,
    kernel_size=2,
    strides=1,
    up_size=2,
    padding="same",
    use_bn=False,
    use_bias=True,
    use_dropout=False,
    drop_value=_DROPOUT_DEFAULT,
):
    if use_bn:
        use_bias=False
    if up_size > 1: x = l.UpSampling1D(up_size)(x)
    x = l.Conv1D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)
    if use_bn:
        x = l.BatchNormalization()(x)
    if activation:
        x = activation(x)
    if use_dropout:
        x = l.Dropout(drop_value)(x)
    return x


def conv_block(
    x,
    filters,
    activation,
    kernel_size=2,
    strides=2,
    padding="same",
    use_bias=True,
    use_bn=False,
    use_dropout=False,
    drop_value=_DROPOUT_DEFAULT,
):
    x = l.Conv1D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)
    if use_bn:
        x = l.BatchNormalization()(x)
    x = activation(x)
    if use_dropout:
        x = l.Dropout(drop_value)(x)
    return x

def get_ae(nInputs):
    inLayer = l.Input(shape=(NTIMESTEPS, nInputs))
    featSizeFactor = 1.4
    featSize = nInputs * featSizeFactor
    x = conv_block(inLayer, featSize, l.LeakyReLU(_LEAKY_ALPHA_DEFAULT))
    for i in range(2):
        featSize *= featSizeFactor
        x = conv_block(x, int(featSize), l.LeakyReLU(_LEAKY_ALPHA_DEFAULT))
    featSize *= featSizeFactor
    latent = conv_block(x, int(featSize), l.LeakyReLU(_LEAKY_ALPHA_DEFAULT))
    featSize /= featSizeFactor
    x = upsample_block(latent, int(featSize), l.LeakyReLU(_LEAKY_ALPHA_DEFAULT))
    for i in range(2):
        featSize /= featSizeFactor
        x = upsample_block(x, int(featSize), l.LeakyReLU(_LEAKY_ALPHA_DEFAULT))

    featSize /= featSizeFactor
    x = upsample_block(x, int(featSize), keras.activations.sigmoid)
    model = keras.models.Model([inLayer], [x, latent], name="AutoEncoder")
    model.compile(optimizer=_OPTIMIZER, loss=_LOSSFUNC, metrics=_METRICS)
    return model

if __name__ == "__main__":
    # df = get_data()
    model = get_ae(150)
    exit()