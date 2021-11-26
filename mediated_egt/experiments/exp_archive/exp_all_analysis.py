# %%
from evolution import *
import ast
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
from itertools import product, combinations
from functools import reduce
from dataframe import makeEntry2, makeColumns
import seaborn as sns
import pandas as pd

# %%

# %%
# loading results from date run
# res = loadExperimentDf("/mnt/c/Users/frsc/Documents/Projects/aligning_rs/graphEgtExperiments/self_interested_mediators/experiments/data/analyzed/jul-2")
# path = "/mnt/c/Users/frsc/Documents/Projects/aligning_rs/graphEgtExperiments/self_interested_mediators/experiments/data/jul7_most_complete_to_date"
path = "/mnt/c/Users/frsc/Documents/Projects/aligning_rs/graphEgtExperiments/self_interested_mediators/experiments/data/jul23"
res = loadExperimentDf(path)
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
# loading results from df
# path = "/mnt/c/Users/frsc/Documents/Projects/aligning_rs/graphEgtExperiments/self_interested_mediators/experiments/data/jul7_most_complete_to_date/jul7.csv"
# path = "/mnt/c/Users/frsc/Documents/Projects/aligning_rs/graphEgtExperiments/self_interested_mediators/experiments/data/jul16/jul16_singles_df.csv"
path = "/mnt/c/Users/frsc/Documents/Projects/aligning_rs/graphEgtExperiments/self_interested_mediators/experiments/data/jul23/jul23_small_init.csv"
# path = "/mnt/c/Users/frsc/Documents/Projects/aligning_rs/graphEgtExperiments/self_interested_mediators/experiments/data/jul7_most_complete_to_date/jul7.csv"
converter_tuples = {"meds": ast.literal_eval,
                    "ts": ast.literal_eval, "ws": lambda x: eval(x)}
# converter_tuples = {"meds": ast.literal_eval, "ws": ast.literal_eval, "ts": ast.literal_eval}
df = pd.read_csv(path, header=[0, 1], index_col=0, converters=converter_tuples)
df = df.round(3)
df = df.rename(columns={'Unnamed: 44_level_1': '',
               'Unnamed: 45_level_1': '', 'Unnamed: 46_level_1': ''}, level=1)
df_mean = df.groupby(["meds", "ws"]).mean()
df_mean_ts = df.groupby(["meds", ("params", "W1"), "ts"]).mean()


# %%
single_100k_df = df[df['meds'].apply(
    lambda x: len(x) == 1) & (df['meds'] != ("NO_MED",))]
df = single_100k_df[single_100k_df[("params", "W1")] != inf]
df_mean = df.groupby(["meds", "ws"]).mean()
df_mean_ts = df.groupby(["meds", ("params", "W1"), "ts"]).mean()

# %%
# plot many
w1s = list(pd.unique(df_mean_ts.index.get_level_values(1)))
medSets = list(pd.unique(df_mean_ts.index.get_level_values(0)))
singleMedSets = [m for m in medSets if len(m) == 1]
single_w1s = w1s

for m in singleMedSets:
    for w1 in single_w1s:
        plot_ts(w1, m)

# %%
# big grid for heatmap for coop/heterogeneity/etc

# %%
# plot 5 heatmaps for each mediator set + 1 barplot per w1,w2 combo


def plot_heatmap(df, vmin, vmax, yticklabels, xticklabels, title='', ax=None, interpolate=True):
    if not interpolate:
        return sns.heatmap(df, vmin=vmin, vmax=vmax, yticklabels=yticklabels, xticklabels=xticklabels, annot=True, square=True, ax=ax).set(title=title)
    else:
        if not ax:
            print("not ax")
            ax = plt.gca()
        im = ax.imshow(df, interpolation='spline36', vmin=vmin, vmax=vmax)
        # im = ax.imshow(df, interpolation='lanczos', vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_xticklabels(xticklabels)
        ax.set_yticklabels(yticklabels)
        ax.set_title(title)
        cbar = ax.figure.colorbar(im, ax=ax)
        # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        return im


def plot_competition(medSet, df):
    interpolate = False
    df_mean = df.groupby(["meds", "ws"]).mean()
    n = 5
    size = 5
    fig, ax = plt.subplots(1, n, figsize=((n+1)*size, size))
    df_pivoted = df_mean.loc[medSet].pivot(
        index=[("params", "W1")], columns=[("params", "W2")])
    w1s, w2s = [round(x, 3) for x in df_pivoted[("agents", "coop_freq")].index], [
        round(x, 3) for x in df_pivoted[("agents", "coop_freq")].columns]
    plot_heatmap(df_pivoted[("agents", "coop_freq")], vmin=0, vmax=1, yticklabels=w1s,
                 xticklabels=w2s, title=f"Avg. final #cooperators", ax=ax[0], interpolate=interpolate)
    plot_heatmap(df_pivoted[("net", "heterogeneity")], vmin=df_mean[('net', 'heterogeneity')].min(), vmax=df_mean[('net', 'heterogeneity')].max(
    ), yticklabels=w1s, xticklabels=w2s, title=f"Avg. final degree heterogeneity", ax=ax[1], interpolate=interpolate)
    plot_heatmap(df_pivoted[("net", "k_max")], vmin=df_mean[('net', 'k_max')].min(), vmax=df_mean[('net', 'k_max')].max(
    ), yticklabels=w1s, xticklabels=w2s, title=f"Avg. final max degree", ax=ax[2], interpolate=interpolate)
    # rewire per opportunity: rewire numbers adjusted to stop time and W parameters.
    df_piv_rewire_n = (df_pivoted[("net", "rewire_n")]).divide(1-1/(df_pivoted[("net", "rewire_n")].index+1),
                                                               axis=0) / 1/(df_pivoted[("net", "rewire_n")].columns+1) / df_pivoted[("net", "stop_n")]
    plot_heatmap(df_piv_rewire_n, vmin=0, vmax=1, yticklabels=w1s, xticklabels=w2s,
                 title=f"Avg. final #rewires / opportunity", ax=ax[3], interpolate=interpolate)
    plot_heatmap(df_pivoted[("net", "stop_n")], vmin=df_mean[('net', 'stop_n')].min(), vmax=df_mean[('net', 'stop_n')].max(
    ), yticklabels=w1s, xticklabels=w2s, title=f"Avg. final stop time", ax=ax[4], interpolate=interpolate)
    plt.suptitle(f"Mediators: {medSet}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# plot a row of heatmaps for a given mediator set and w1
def plot_ts_row(w1, medSet, df_mean_ts, axs=None):
    interpolate = True

    def plot_ts_row_w_axis(axs):
        ax = axs
        df_pivoted = df_mean_ts.loc[medSet].loc[w1].pivot(
            index=[("game", "s")], columns=[("game", "t")]).sort_index(ascending=False)
        ss, ts = ['%.2f' % elem for elem in df_pivoted[("agents", "coop_freq")].index], [
            '%.2f' % elem for elem in df_pivoted[("agents", "coop_freq")].columns]
        plot_heatmap(df_pivoted[("agents", "coop_freq")], vmin=0, vmax=1, yticklabels=ss,
                     xticklabels=ts, title=f"Avg. final #cooperators", ax=ax[0], interpolate=interpolate)
        plot_heatmap(df_pivoted[("net", "heterogeneity")], vmin=df_mean_ts[('net', 'heterogeneity')].min(), vmax=df_mean_ts[('net', 'heterogeneity')].max(
        ), yticklabels=ss, xticklabels=ts, title=f"Avg. final degree heterogeneity", ax=ax[1], interpolate=interpolate)
        plot_heatmap(df_pivoted[("net", "k_max")], vmin=df_mean_ts[('net', 'k_max')].min(), vmax=df_mean_ts[('net', 'k_max')].max(
        ), yticklabels=ss, xticklabels=ts, title=f"Avg. final max degree", ax=ax[2], interpolate=interpolate)
        # rewire per opportunity: rewire numbers adjusted to stop time and W parameters.
        df_piv_rewire_n = (df_pivoted[("net", "rewire_n")]) / \
            w1 / df_pivoted[("net", "stop_n")]
        plot_heatmap(df_piv_rewire_n, vmin=0, vmax=1, yticklabels=ss, xticklabels=ts,
                     title=f"Avg. final #rewires / opportunity", ax=ax[3], interpolate=interpolate)
        plot_heatmap(df_pivoted[("net", "stop_n")], vmin=df_mean_ts[('net', 'stop_n')].min(), vmax=df_mean_ts[('net', 'stop_n')].max(
        ), yticklabels=ss, xticklabels=ts, title=f"Avg. final stop time", ax=ax[4], interpolate=interpolate)
    if axs != None and axs.all() != None:
        # if axs.all() != None:
        plot_ts_row_w_axis(axs)
    else:
        size = 5
        n = 5
        fig, new_axs = plt.subplots(1, n, figsize=(
            (n+1)*size, size), constrained_layout=False)
        plt.suptitle(f"Mediators: {medSet}, w1={w1}")
        df_mean_ts = df.groupby(["meds", ("params", "W1"), "ts"]).mean()
        plot_ts_row_w_axis(new_axs)
        fig.tight_layout(rect=[0, 0, 1, 0.95])


# plot ts for a given set of mediators and each w1
def plot_ts(medSet, df):
    filter_inf = True
    df = df if not filter_inf else df[df[("params", "W1")] != inf]
    n = 5
    size = 5
    df_mean_ts = df.groupby(["meds", ("params", "W1"), "ts"]).mean()
    w1s = df_mean_ts.index.levels[1]
    fig, axs = plt.subplots(len(w1s), n, figsize=(
        (n+1)*size, len(w1s) * size), constrained_layout=False)
    for i, w1 in enumerate(w1s):
        plot_ts_row(w1, medSet, df_mean_ts, axs[i])
    plt.suptitle(f"Mediators: {medSet}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig
# Makes a pandas bar plot of mediator frequencies for each w parameter combination


def plot_bar_grid(medSet, df):
    df_mean = df.groupby(["meds", "ws"]).mean()
    size = 5
    w1s = [round(x, 3)
           for x in pd.unique(df_mean.loc[medSet][("params", "W1")])]
    w2s = [round(x, 3)
           for x in pd.unique(df_mean.loc[medSet][("params", "W2")])]
    fig, axs = plt.subplots(len(w1s), len(
        w2s), figsize=((len(w1s)+1)*size, len(w2s)*size))
    for i, w1 in enumerate(w1s):
        for j, w2 in enumerate(w2s):
            med_freq_ax = df_mean.loc[medSet].loc[[(w1, w2)]]['med_freqs'].plot.bar(
                ax=axs[i, j], xlabel="mediators", ylim=(0, 1), xticks=[], label=str())
            med_freq_ax.set_title(
                f"Mediator freqs w1={w1} w2={w2}", color='black')
            med_freq_ax.legend(bbox_to_anchor=(1.0, 1.0))
            med_freq_ax.plot()
    plt.suptitle(f"Mediator frquency: {medSet}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# Makes a pandas violin plot of mediator frequencies for each w parameter combination


def plot_violin_grid(medSet, df):
    df_mean = df.groupby(["meds", "ws"]).mean()
    size = 5
    w1s = [round(x, 3)
           for x in pd.unique(df_mean.loc[medSet][("params", "W1")])]
    w2s = [round(x, 3)
           for x in pd.unique(df_mean.loc[medSet][("params", "W2")])]
    fig, axs = plt.subplots(len(w1s), len(
        w2s), figsize=((len(w1s)+1)*size, len(w2s)*size))
    plt.suptitle(f"Mediator frquency: {medSet}")
    for i, w1 in enumerate(w1s):
        for j, w2 in enumerate(w2s):
            df_plot = df[(df['meds'] == medSet) & (df['ws'] == (w1, w2))][[
                ("med_freqs", x) for x in medSet]].stack().reset_index()
            plt.figure(figsize=(10, 5))
            ax = sns.violinplot(x="level_1", y="med_freqs",
                                data=df_plot, inner=None, ax=axs[i, j]).set_title(f"Mediator freqs w1={w1} w2={w2}", color='black')
            ax = sns.swarmplot(x="level_1", y="med_freqs", data=df_plot,
                               color="white", edgecolor="gray", size=2,  ax=axs[i, j])
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# %%
ipywidgets.interact(plot_ts,
                    w1=list(pd.unique(df_mean_ts.index.get_level_values(1))),
                    medSet=list(
                        pd.unique(df_mean_ts.index.get_level_values(0))),
                    df=ipywidgets.fixed(df)
                    )

# %%

ipywidgets.interact(plot_competition,
                    medSet=list(pd.unique(df_mean.index.get_level_values(0))),
                    df=ipywidgets.fixed(df)
                    )

# %%
# plot ts plots for each single med and w
for m in pd.unique(df_mean_ts.index.get_level_values(0)):
    fig = plot_ts(m, df)
    dir = "/mnt/c/Users/frsc/Documents/Projects/aligning_rs/graphEgtExperiments/self_interested_mediators/experiments/plots/single_interp_400k"
    fig.savefig(f"{dir}/single_{m}_400k_ep.png")

# %%
# plot violin plots for each 1v1
for m in pd.unique(df_mean.index.get_level_values(0)):
    fig = plot_violin_grid(m, df)
    dir = "/mnt/c/Users/frsc/Documents/Projects/aligning_rs/graphEgtExperiments/self_interested_mediators/experiments/plots/small_init_comp/violins"
    fig.savefig(f"{dir}/{m}_comp.png")

# %%