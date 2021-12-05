#80/20 Hillary graph plots
import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as l
import numpy as np


import util
import constants as gv
import timeDataProcessing as tdp
from model_paths import keras_model_paths

NTIMESTEPS = 16
_DROPOUT_DEFAULT = 0.25
_LEAKY_ALPHA_DEFAULT = 0.2
_OPTIMIZER = keras.optimizers.Adam(2e-4, 0.5)
_LOSSFUNC = keras.losses.MeanSquaredError()
_KERNELINIT = keras.initializers.RandomNormal(stddev=0.02)
_METRICS = [keras.metrics.MeanAbsoluteError(),
            keras.losses.CosineSimilarity(),]

_UPSAMPLE_KW = {"strides": 1, "up_size": 2}
_DOWNSAMPLE_KW = {"strides": 2, "up_size": 1}


def conv_block(
    x,
    filters,
    activation,
    kernel_size=2,
    strides=2,
    up_size=1,
    padding="same",
    use_bn=False,
    use_bias=True,
    use_dropout=False,
    use_max_pool=False,
    use_avg_pool=False,
    drop_value=_DROPOUT_DEFAULT,
):
    if use_bn:
        use_bias=False
    if up_size > 1: x = l.UpSampling1D(up_size)(x)
    x = l.Conv1D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias,
    )(x)
    if use_bn:
        x = l.BatchNormalization()(x)
    if use_max_pool:
        x = l.MaxPooling1D()(x)
    if use_avg_pool:
        x = l.AveragePooling1D()(x)
    if activation:
        x = activation(x)
    if use_dropout:
        x = l.Dropout(drop_value)(x)
    return x


class autoencoder():
    encoderName = "encoder"
    decoderName = "decoder"
    _defaultDir = keras_model_paths.directory.aeModels
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        self.model = keras.Sequential()
        self.model.add(self.encoder)
        self.model.add(self.decoder)
        self.model.compile(optimizer=_OPTIMIZER, loss=_LOSSFUNC, metrics=_METRICS)

    @staticmethod
    def _enc_file_name(dir):
        return dir + autoencoder.encoderName + keras_model_paths.extensions.kerasModel
    @staticmethod
    def _dec_file_name(dir):
        return dir + autoencoder.decoderName + keras_model_paths.extensions.kerasModel

    def save(self, dir = _defaultDir):
        self.encoder.save(autoencoder._enc_file_name(dir))
        self.decoder.save(autoencoder._dec_file_name(dir))

    @staticmethod
    def load_ae(dir = _defaultDir):
        return autoencoder(
            keras.models.load_model(autoencoder._enc_file_name(dir)),
            keras.models.load_model(autoencoder._dec_file_name(dir))
        )

    @staticmethod
    def create_ae(nInputs):
        inLayer = l.Input(shape=(NTIMESTEPS, nInputs))
        featSizeFactor = 1.4
        featSize = nInputs * featSizeFactor
        x = conv_block(inLayer, featSize, l.LeakyReLU(_LEAKY_ALPHA_DEFAULT), **_DOWNSAMPLE_KW)
        for i in range(2):
            featSize *= featSizeFactor
            x = conv_block(x, int(featSize), l.LeakyReLU(_LEAKY_ALPHA_DEFAULT), **_DOWNSAMPLE_KW)
        featSize *= featSizeFactor
        x = conv_block(x, int(featSize), l.LeakyReLU(_LEAKY_ALPHA_DEFAULT), **_DOWNSAMPLE_KW)
        encoder = keras.models.Model(inLayer, x, name=autoencoder.encoderName)

        inLayer = l.Input(shape=(1, int(featSize)))
        featSize /= featSizeFactor
        x = conv_block(inLayer, int(featSize), l.LeakyReLU(_LEAKY_ALPHA_DEFAULT), **_UPSAMPLE_KW)
        for i in range(2):
            featSize /= featSizeFactor
            x = conv_block(x, int(featSize), l.LeakyReLU(_LEAKY_ALPHA_DEFAULT), **_UPSAMPLE_KW)

        featSize /= featSizeFactor
        x = conv_block(x, int(featSize), keras.activations.sigmoid, **_UPSAMPLE_KW)
        decoder = keras.models.Model([inLayer], x, name=autoencoder.decoderName)
        return autoencoder(encoder, decoder)



def train_ae(windows, load):
    stepsPerEpoch = 2 if gv.DEBUG else None
    epochs = 2 if gv.DEBUG else 1
    useValidation=True

    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss" if useValidation else "loss", patience=0)]
    ae = autoencoder.load_ae() if load else autoencoder.create_ae(windows.shape[-1])
    ae.model.fit(
        windows, windows, validation_split=.25 if useValidation else 0., callbacks=callbacks,
        steps_per_epoch=stepsPerEpoch, epochs=epochs
    )

    if not gv.DEBUG:
        ae.save()
    return ae



if __name__ == "__main__":
    nw = tdp.network_window.get_window_data(NTIMESTEPS)
    benign = nw.get_homogeneous_benign()
    hetero = nw.get_only_heterogeneous()
    ae = train_ae(benign.windows, load=False)

    exit()