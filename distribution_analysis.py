import math

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest

import autoencoder
import file_paths as fp
import constants as gv
import timeDataProcessing as tdp
import autoencoder as autoenc

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

def line_plot_n_malicious(
        ae:autoencoder.autoencoder,
        networkWindow:tdp.network_window,
        metric, nRange = (0,autoenc.NTIMESTEPS), name="", ylabel=""
):
    vals = []
    for i in range(*nRange):
        # print("Number of malicious", str(i) + "/" + str(nRange[1]))
        vals.append(
            float(ae.evaluate(networkWindow.get_n_malicious(i), metric))
        )
        # print(vals[-1])
    xs = [i/nRange[1] for i in range(*nRange)]
    ax = sns.lineplot(xs, vals)
    ax.set_title(name)
    ax.set_xlabel("Fraction of Malicious Packets")
    ax.set_ylabel(ylabel)
    # sns.despine(ax=ax, top=False, right=False)
    if not gv.DEBUG:
        plt.savefig(fp.images.directory.distributionAnalysis + name.replace(" ", "") + ".png")
    plt.figure()
    return vals

def plot_n_mal_sizes():
    sizes = [173685, 49750, 21443, 12050, 8816, 7184, 6441,5710,5482,4836,4575,3903,3514,2973,2377,2423,2342]
    fractions = [x/16. for x in range(len(sizes))]
    plt.figure()
    ax = sns.lineplot(fractions,sizes)
    ax.set_title("Data Amount Given Malicous Packet Fracton")
    ax.set_ylabel("Window Count")
    ax.set_xlabel("Fraction of Malicious Packets")
    plt.subplots_adjust(left=.14)
    plt.savefig(fp.images.directory.distributionAnalysis + "dataSizeLine.png")
    return ax

def get_losses_per_batch(
        ae:autoencoder.autoencoder,
        arr3d:np.ndarray,
        nGroups, metricName:str = "loss"
):
    step = arr3d.shape[0]/nGroups
    print("Doing %d steps and %d groups for %d samples." %(step, nGroups, arr3d.shape[0]))
    end=0.
    losses = []
    for i in range(0,nGroups):
        pivot1 = int(end)
        end += step
        pivot = min(int(end), arr3d.shape[0])
        # pivot = i+1
        metrics = ae.model.evaluate(arr3d[pivot1:pivot], arr3d[pivot1:pivot], return_dict=True, verbose=0)
        losses.append(metrics[metricName])
    return losses

def get_lossDf(
        ae: autoencoder.autoencoder,
        networkWindow: tdp.network_window,
        metric="loss", nGroups=20 if gv.DEBUG else 100, nMaliciouses=None
    ):
    if nMaliciouses is None:
        nMaliciouses = [0,8,16] if gv.DEBUG else [i for i in range(0,17)] # [0,4,8,12,16]
    df = {}
    for nMalicious in nMaliciouses:
        print("%s malicious packets" %(nMalicious))
        df[nMalicious] = get_losses_per_batch(ae, networkWindow.get_n_malicious(nMalicious), nGroups, metricName=metric)

    df = pd.DataFrame(df)
    return df

def dist_plots(lossDf:pd.DataFrame, suffix="", ylabel="",
):
    lossDf.sort_index(inplace=True)
    ax = sns.displot(lossDf, kind="hist", multiple='stack')
    plt.figure()
    ax = sns.displot(lossDf, kind="kde", fill=True, legend="# malicious", ax=ax, height=4, aspect=2)
    ax.set_xlabels("Cosine Similarity")
    # ax.suptitle("Distribution of Frame (Size 16) With Number of Malicious Packets")
    ax.set(xlim=(-1,0.) if suffix=="PCA" else (-1,-0.9))
    plt.title("Loss Given the Number of Malicious Packets using " + suffix)
    plt.subplots_adjust(top=.9, bottom=.15)
    # plt.legend(nMaliciouses, "Number of Malicious Packets")
    # ax.set_xlabel("Cosine Similarity")
    # sns.despine(ax=ax, top=False, right=False)
    if not gv.DEBUG:
        plt.savefig(fp.images.directory.distributionAnalysis + "KdeLoss" + suffix.replace(' ','') + ".png")

    return ax

class bucket:
    def __init__(self, start, stop, count=0):
        self.start = start
        self.stop = stop
        self.count =  count
        self.rng = (start,stop)

def get_bucket_dict(start, stop, step):
    return {key: bucket(start, start + step) for key in np.arange(start,stop,step)}


def bucket_probs(
        lossDf:pd.DataFrame, suffix="", rngArgs = (-1,1.00001,.01)
):
    if rngArgs is None:
        rngArgs = (-1, 1.00001, .01)
    if suffix is None:
        suffix = ""
    dfCounts = pd.DataFrame()
    bins = np.arange(*rngArgs)
    for col in lossDf.columns:
        dfCounts[col] = pd.cut(lossDf[col], bins=bins, retbins=True)[0].value_counts()

    sums = dfCounts.sum(axis=1)
    sums = sums[sums > 2]
    normedDf = dfCounts.loc[sums.index].div(sums, axis=0).dropna()
    normedDf.sort_index(inplace=True)

    fig = plt.figure(figsize=(9.5,6), **_defPlotKwArgs)
    ax = sns.heatmap(normedDf)
    ax.set_title("Heatmap for N/16 Malicious Packets using " + suffix)
    ax.set_ylabel("Cosine Similarity Range")
    ax.set_xlabel("Number of Malicious Packets")

    name = "cosSimHeatMapNormedProb" + suffix
    if not gv.DEBUG:
        plt.savefig(fp.images.directory.distributionAnalysis + name.replace(' ','') + ".png")

    colSize = len(normedDf.columns)
    pivot = math.ceil(colSize / 2)

    normedDf[normedDf.columns[:pivot]].to_latex(fp.miscDir + name + ".tex", index=True, float_format="%.3f")
    normedDf[normedDf.columns[pivot:]].to_latex(fp.miscDir + name + "rest.tex", index=True, float_format="%.3f")
    return normedDf