import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest


import file_paths as fp
import constants as gv

_defaultDir = fp.images.directory.distributionAnalysis

_defPlotKwArgs = {"tight_layout":True}

def kolm_smirnov_analysis(data1, data2, nameSuffix):
    length = 10
    size = (length,length)
    plotArgs = {"tight_layout":True, "figsize":size}

    kssWindow, pvalsWindow = kolm_smirnov_by_window(data1, data2)
    kssFeatures = kssWindow.mean(axis=1)
    pvalsFeatures = pvalsWindow.mean(axis=1)
    fig = plt.figure(**plotArgs)
    cmap = sns.cubehelix_palette(as_cmap=True)
    ax = sns.kdeplot(kssFeatures, pvalsFeatures, cmap=cmap, fill=True)
    ax.set_title("KS Scores and Respective P-Values for " + nameSuffix)
    ax.set_xlabel("Kolmogorov-Smirnov Scores")
    ax.set_ylabel("Two-Sided P-Values")
    if not gv.DEBUG:
        plt.savefig(_defaultDir + nameSuffix.replace(' ','') + "Kde.png")

    fig = plt.figure(tight_layout=True, figsize=(10,7))
    ax = sns.boxplot(kssFeatures)
    ax.set_title("Kolmogorov-Smirnov Scores for " + nameSuffix)
    if not gv.DEBUG:
        plt.savefig(_defaultDir + nameSuffix.replace(' ','') + "Boxplot.png")

    def round(val):
        return str(np.around(val, decimals=3))

    if gv.DEBUG:
        print(nameSuffix, "KS Mean", round(kssFeatures.mean()), "STD", round(kssFeatures.std()))
        print(nameSuffix, "P-vals Mean", round(pvalsFeatures.mean()), "STD", round(pvalsFeatures.std()))
    colDelim = " & "
    global _fileOutBuffer
    _fileOutBuffer += nameSuffix + colDelim + round(kssFeatures.mean()) + colDelim + round(kssFeatures.std()) \
                      + colDelim + round(pvalsFeatures.mean()) + colDelim + round(pvalsFeatures.std()) + '\\\\\n'
    return kssWindow, pvalsWindow

def kolm_smirnov_by_feature(data1, data2):
    assert data1.shape == data2.shape
    nFeatures = data1.shape[-1]
    kss, pvals = np.zeros(shape=(nFeatures)), np.zeros(shape=(nFeatures))
    for i in range(nFeatures):
        kss[i], pvals[i] = kstest(data1[..., i].flatten(), data2[..., i].flatten())
    return kss, pvals

def kolm_smirnov_by_window(data1, data2):
    assert data1.shape == data2.shape
    nSamples = data1.shape[0]
    nFeatures = data1.shape[-1]
    kss = np.zeros(shape=(nSamples, nFeatures))
    pvals = np.zeros(shape=(nSamples, nFeatures))
    for i in range(nSamples):
        kss[i], pvals[i] = kolm_smirnov_by_feature(data1[i], data2[i])
        # if meta.DEBUG:
        #     break
    return kss, pvals


def quantitative_analyses(windows1, windows2, name):
    kmStats = kolm_smirnov_analysis(windows1, windows2, name)
    return kmStats

