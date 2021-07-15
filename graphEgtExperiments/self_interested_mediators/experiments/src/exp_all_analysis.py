# %%
import ipywidgets
import matplotlib.pyplot as plt
from itertools import chain
from utils import transposeList
from plots import *
from egt_io import *
from defaultParams import _episode_n, _N, _W, _W2, _k, _T, _S
from defaultParams import *
from mediators import *
from mediators import _medSet
from evolution import *
from itertools import product, combinations
from functools import reduce
from dataframe import makeEntry2, makeColumns
import seaborn as sns
import pandas as pd


# %%
# loading results from jul2
# res = loadExperimentDf("/mnt/c/Users/frsc/Documents/Projects/aligning_rs/graphEgtExperiments/self_interested_mediators/experiments/data/analyzed/jul-2")
res = loadExperimentDf(
    "/mnt/c/Users/frsc/Documents/Projects/aligning_rs/graphEgtExperiments/self_interested_mediators/experiments/data/jul7_most_complete_to_date")
df = pd.concat(res, ignore_index=True)
df = df.round(3)
df[[('game', 't'), ('game', 's')]] = df[[('game', 's'), ('game', 't')]]
df['meds'] = df['med'].apply(lambda r: tuple(
    c for c in df['med'].columns if r[c] == 1), axis=1)  # tuples of mediators
# tuples of ws pairs
df['ws'] = list(zip(df[('params', 'W1')], df[('params', 'W2')]))
df['ts'] = list(zip(df[('game', 't')], df[('game', 's')]))
df_mean = df.groupby(["meds", "ws"]).mean()
df_mean_ts = df.groupby(["meds", ("params", "W1"), "ts"]).mean()
# %%
# big grid for heatmap for coop/heterogeneity/etc

# %%
# plot 5 heatmaps for each mediator set + 1 barplot per w1,w2 combo


def plot_bar(w1, w2, medSet):
    n = 6
    size = 5
    fig, ax = plt.subplots(1, n, figsize=((n+1)*size, size))
    plt.suptitle(f"Mediators: {medSet}, w1={w1}, w2={w2}")
    med_freq_ax = df_mean.loc[[(medSet, (w1, w2))]]['med_freqs'].plot.bar(
        ax=ax[0], xlabel="mediators", ylim=(0, 1), xticks=[], label=str())
    med_freq_ax.set_title("Mediator freqs", color='black')
    med_freq_ax.legend(bbox_to_anchor=(1.0, 1.0))
    med_freq_ax.plot()
    df_pivoted = df_mean.loc[medSet].pivot(
        index=[("params", "W1")], columns=[("params", "W2")])
    w1s, w2s = list(df_pivoted[("agents", "coop_freq")].index), list(
        df_pivoted[("agents", "coop_freq")].columns)
    sns.heatmap(df_pivoted[("agents", "coop_freq")], vmin=0, vmax=1, yticklabels=w1s,
                xticklabels=w2s, annot=True, square=True, ax=ax[1], ).set(title=f"Avg. final #cooperators")
    sns.heatmap(df_pivoted[("net", "heterogeneity")], vmin=df_mean[('net', 'heterogeneity')].min(), vmax=df_mean[('net', 'heterogeneity')].max(
    ), yticklabels=w1s, xticklabels=w2s, annot=True, square=True, ax=ax[2]).set(title=f"Avg. final degree heterogeneity")
    sns.heatmap(df_pivoted[("net", "k_max")], vmin=df_mean[('net', 'k_max')].min(), vmax=df_mean[('net', 'k_max')].max(
    ), yticklabels=w1s, xticklabels=w2s, annot=True, square=True, ax=ax[3]).set(title=f"Avg. final max degree")
    sns.heatmap(df_pivoted[("net", "rewire_n")], vmin=df_mean[('net', 'rewire_n')].min(), vmax=df_mean[('net', 'rewire_n')].max(
    ), yticklabels=w1s, xticklabels=w2s, annot=True, square=True, ax=ax[4]).set(title=f"Avg. final #rewires")
    sns.heatmap(df_pivoted[("net", "stop_n")], vmin=df_mean[('net', 'stop_n')].min(), vmax=df_mean[('net', 'stop_n')].max(
    ), yticklabels=w1s, xticklabels=w2s, annot=True, square=True, ax=ax[5]).set(title=f"Avg. final stop time")
    fig.tight_layout()


def plot_ts(w1, medSet):
    n = 5
    size = 5
    fig, ax = plt.subplots(1, n, figsize=(
        (n+1)*size, size), constrained_layout=False)
    plt.suptitle(f"Mediators: {medSet}, w1={w1}")
    # df_mean_ts = df.groupby(["meds", "ts"]).mean()
    df_pivoted = df_mean_ts.loc[medSet].loc[w1].pivot(
        index=[("game", "s")], columns=[("game", "t")]).sort_index(ascending=False)
    ss, ts = ['%.2f' % elem for elem in df_pivoted[("agents", "coop_freq")].index], [
        '%.2f' % elem for elem in df_pivoted[("agents", "coop_freq")].columns]
    sns.heatmap(df_pivoted[("agents", "coop_freq")], vmin=0, vmax=1, yticklabels=ss,
                xticklabels=ts, annot=True, square=True, ax=ax[0], ).set(title=f"Avg. final #cooperators")
    sns.heatmap(df_pivoted[("net", "heterogeneity")], vmin=df_mean_ts[('net', 'heterogeneity')].min(), vmax=df_mean_ts[('net', 'heterogeneity')].max(
    ), yticklabels=ss, xticklabels=ts, annot=True, square=True, ax=ax[1]).set(title=f"Avg. final degree heterogeneity")
    sns.heatmap(df_pivoted[("net", "k_max")], vmin=df_mean_ts[('net', 'k_max')].min(), vmax=df_mean_ts[('net', 'k_max')].max(
    ), yticklabels=ss, xticklabels=ts, annot=True, square=True, ax=ax[2]).set(title=f"Avg. final max degree")
    sns.heatmap(df_pivoted[("net", "rewire_n")], vmin=df_mean_ts[('net', 'rewire_n')].min(), vmax=df_mean_ts[('net', 'rewire_n')].max(
    ), yticklabels=ss, xticklabels=ts, annot=True, square=True, ax=ax[3]).set(title=f"Avg. final #rewires")
    sns.heatmap(df_pivoted[("net", "stop_n")], vmin=df_mean_ts[('net', 'stop_n')].min(), vmax=df_mean_ts[('net', 'stop_n')].max(
    ), yticklabels=ss, xticklabels=ts, annot=True, square=True, ax=ax[4]).set(title=f"Avg. final stop time")
    # fig.tight_layout()


# %%
ipywidgets.interact(plot_ts,
                    w1=list(pd.unique(df_mean_ts.index.get_level_values(1))),
                    medSet=list(
                        pd.unique(df_mean_ts.index.get_level_values(0)))
                    )

# %%

ipywidgets.interact(plot_bar,
                    w1=list(pd.unique(df_mean[("params", "W1")])),
                    w2=list(pd.unique(df_mean[("params", "W2")])),
                    medSet=list(pd.unique(df_mean.index.get_level_values(0)))
                    )

# %%
