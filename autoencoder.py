#80/20 Hillary graph plots
import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as l
import numpy as np
import matplotlib.pyplot as plt
import copy
from enum import Enum

import constants
import util
import constants as gv
import timeDataProcessing as tdp
from file_paths import keras_model_paths
import distribution_analysis as da

NTIMESTEPS = 16
_DROPOUT_DEFAULT = 0.25
_LEAKY_ALPHA_DEFAULT = 0.2
_OPTIMIZER = keras.optimizers.Adam(2e-4, 0.5)
# _LOSSFUNC = keras.losses.MeanSquaredError()
_LOSSFUNC = keras.losses.CosineSimilarity()
_KERNELINIT = keras.initializers.RandomNormal(stddev=0.02)
_METRICS = [keras.metrics.MeanAbsoluteError(),
            keras.losses.CosineSimilarity(),]
_NDATA = 100000 if gv.DEBUG else None

_UPSAMPLE_KW = {"strides": 1, "up_size": 2}
_DOWNSAMPLE_KW = {"strides": 2, "up_size": 1}

class spec_nn(Enum):
    PCA = 0
    FULL = 1

specificNnSuffix = {
    spec_nn.PCA : "PCA",
    spec_nn.FULL : "All Dimensions"
}

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
        use_bias=False #batch normalization internalizes the bias
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
    _defaultDir = keras_model_paths.directory.autoencoderModels
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        self.model = keras.Sequential()
        self.model.add(self.encoder)
        self.model.add(self.decoder)
        self.model.compile(optimizer=_OPTIMIZER, loss=_LOSSFUNC, metrics=_METRICS)

    def evaluate(self, data, func):
        pred = self.model(data)
        return func(data, pred)

    @staticmethod
    def _enc_file_name(suffix, dir):
        return dir + autoencoder.encoderName + suffix + keras_model_paths.extensions.kerasModel
    @staticmethod
    def _dec_file_name(suffix, dir):
        return dir + autoencoder.decoderName + suffix + keras_model_paths.extensions.kerasModel

    def save(self,suffix,  dir = _defaultDir):
        self.encoder.save(autoencoder._enc_file_name(suffix, dir))
        self.decoder.save(autoencoder._dec_file_name(suffix, dir))

    @staticmethod
    def load_ae(suffix, dir = _defaultDir):
        return autoencoder(
            keras.models.load_model(autoencoder._enc_file_name(suffix, dir)),
            keras.models.load_model(autoencoder._dec_file_name(suffix, dir))
        )

    @staticmethod
    def create_ae(nInputs, featSizeFactor):
        aeLayerDefArgs = {"use_bn":True}

        inLayer = l.Input(shape=(NTIMESTEPS, nInputs))
        featSizesDec = [nInputs]
        for i in range(4):
            featSizesDec.append(np.round(featSizesDec[-1]*featSizeFactor).astype(int))
        featSizesEnc = copy.deepcopy(featSizesDec)[::-1]
        featSizesEnc.pop()

        x = conv_block(inLayer, featSizesEnc.pop(), l.LeakyReLU(_LEAKY_ALPHA_DEFAULT),
                       **_DOWNSAMPLE_KW, **aeLayerDefArgs,)
        for i in range(2):
            x = conv_block(x, featSizesEnc.pop(), l.LeakyReLU(_LEAKY_ALPHA_DEFAULT),
                           **_DOWNSAMPLE_KW, **aeLayerDefArgs,)

        x = conv_block(x, featSizesEnc.pop(), keras.activations.sigmoid, **_DOWNSAMPLE_KW, **aeLayerDefArgs,)
        encoder = keras.models.Model(inLayer, x, name=autoencoder.encoderName)

        inLayer = l.Input(shape=(1, featSizesDec.pop()))
        x = conv_block(inLayer, featSizesDec.pop(), l.LeakyReLU(_LEAKY_ALPHA_DEFAULT), **_UPSAMPLE_KW, **aeLayerDefArgs,)
        for i in range(2):
            x = conv_block(x, featSizesDec.pop(), l.LeakyReLU(_LEAKY_ALPHA_DEFAULT), **_UPSAMPLE_KW, **aeLayerDefArgs,)

        x = conv_block(x, featSizesDec.pop(), keras.activations.sigmoid, **_UPSAMPLE_KW)
        decoder = keras.models.Model([inLayer], x, name=autoencoder.decoderName)
        print(encoder.summary())
        print(decoder.summary())
        return autoencoder(encoder, decoder)


def train_ae(ae:autoencoder, windows, suffix):
    stepsPerEpoch = 2 if gv.DEBUG else None
    epochs = 2 if gv.DEBUG else 50
    useValidation=True

    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss" if useValidation else "loss", patience=3)]
    ae.model.fit(
        windows, windows, validation_split=.25 if useValidation else 0., callbacks=callbacks,
        steps_per_epoch=stepsPerEpoch, epochs=epochs
    )

    if not gv.DEBUG:
        ae.save(suffix)
    return ae

def benign_malicious_latent(ae:autoencoder, benign, hetero):
    benignOut = ae.encoder(benign.windows).numpy()
    heteroOut = ae.encoder(hetero.windows).numpy()
    # da.kolm_smirnov_analysis(benignOut, heteroOut, "Benign versus Malicious")
    return

def analyze(nw:tdp.network_window, suffix, fit, load, featSizeFactor, bucketsRngArgs):
    ae = autoencoder.load_ae(suffix) if load else autoencoder.create_ae(nw.windows.shape[-1], featSizeFactor)
    print(ae.encoder.summary())
    print(ae.decoder.summary())
    if fit: ae = train_ae(ae, nw.get_n_malicious(0), suffix)

    # benign_malicious_latent(ae, benign, hetero)
    metric = keras.losses.CosineSimilarity()
    # da.line_plot_n_malicious(ae, nw, metric, ylabel="Cosine Similarity", name="Decoded Analysis")
    lossDf = da.get_lossDf(ae, nw,)
    da.bucket_probs(lossDf, suffix=suffix, rngArgs=bucketsRngArgs,)
    # da.dist_plots(lossDf)
    return ae

if __name__ == "__main__":
    if gv.DEBUG: gv.enable_tf_debug()
    nw = tdp.network_window.get_window_data(NTIMESTEPS, firstN=_NDATA)
    load = True
    # load = False
    fit = False
    # fit=True
    # ae = analyze(nw, specificNnSuffix[spec_nn.FULL], fit, load, featSizeFactor=.68, bucketsRngArgs=None)
    # nw.perform_pca()
    ae = analyze(nw, specificNnSuffix[spec_nn.PCA], fit, load, featSizeFactor=1., bucketsRngArgs=(-1,-.6999,.01))

    plt.show()
    exit()