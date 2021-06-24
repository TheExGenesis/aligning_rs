# %%
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
from dataframe import makeEntry2
import seaborn as sns


# %%
# loading
# def plotting_df():
res = loadExperimentDf(
    "/mnt/c/Users/frsc/Documents/Projects/aligning_rs/graphEgtExperiments/self_interested_mediators/experiments/data/exclusive_3_med_competition_jun-22-2021")
res = [r for run in res for r in run]
df = pd.DataFrame(res, columns=makeColumns()).fillna(0)
df_means = df.groupby([*[x for x in df.columns if x[0] == "med"],
                       ("params", "W1"), ("params", "W2")]).mean()
df_means["med_freqs"].iloc[0].plot.bar()  # 1barplot
# select by ws
df_means["med_freqs"][(df_means["med_freqs"].index.get_level_values(('params', 'W1')) == 0.5) & (
    df_means["med_freqs"].index.get_level_values(('params', 'W2')) == 0.5)]
# no_med as base
df_means["med_freqs"][df_means["med_freqs"].index.get_level_values(
    ('med', 'NO_MED')) == 1.0].plot.bar()
# random as base
df_means["med_freqs"][df_means["med_freqs"].index.get_level_values(
    ('med', 'RANDOM_MED')) == 1.0].plot.bar()
df_by_ws_nomed = df_means["med_freqs"][(df_means["med_freqs"].index.get_level_values(('params', 'W1')) == 0.5) & (
    df_means["med_freqs"].index.get_level_values(('params', 'W2')) == 0.5) & (df_means["med_freqs"].index.get_level_values(('med', 'NO_MED')) == 1.0)]
# adding columns
df['meds'] = df['med'].apply(lambda r: tuple(
    c for c in df['med'].columns if r[c] == 1), axis=1)  # tuples of mediators
# tuples of ws pairs
df['ws'] = list(zip(df[('params', 'W1')], df[('params', 'W2')]))
df_mean = df.groupby(["meds", "ws"]).mean()
# barplot grid
df_freqs = df_means['med_freqs'].reset_index()
tidy = df_freqs.melt(id_vars=['ws', 'meds'], value_vars=df['med'].columns)
sns.catplot(x='meds', y='value', hue='variable', col="ws",
            data=tidy, kind="bar", col_wrap=4, aspect=4)

# %%
# big grid for heatmap for coop/heterogeneity/etc


def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    sns.heatmap(d, **kwargs)


df = df.groupby([*[x for x in df.columns if x[0] == "med"],
                 ("params", "W1"), ("params", "W2")]).mean().reset_index()
df['meds'] = df['med'].apply(lambda r: tuple(
    c for c in df['med'].columns if r[c] == 1), axis=1)

# coop
df2 = df.melt(id_vars=["meds"]+[x for x in df.columns if x[0] ==
              "params"], value_vars=[("agents", "coop_freq")], value_name='coop')
df2['W1'] = df2[('params', 'W1')]
df2['W2'] = df2[('params', 'W2')]
fg = sns.FacetGrid(df2, col='meds', col_wrap=4, aspect=1.5)
fg.map_dataframe(draw_heatmap, "W1", "W2", 'coop',
                 cbar=True, square=True, vmin=0, vmax=1)

# heterogeneity
df2 = df.melt(id_vars=["meds"]+[x for x in df.columns if x[0] == "params"],
              value_vars=[("net", "heterogeneity")], value_name='heterogeneity')
df2['W1'] = df2[('params', 'W1')]
df2['W2'] = df2[('params', 'W2')]
fg = sns.FacetGrid(df2, col='meds', col_wrap=4, aspect=1.5)
fg.map_dataframe(draw_heatmap, "W1", "W2", 'heterogeneity',
                 cbar=True, square=True, vmin=0, vmax=df2['heterogeneity'].max())
fg.fig.get_axes()[0].invert_yaxis()  # %%

# k_max
df2 = df.melt(id_vars=["meds"]+[x for x in df.columns if x[0] ==
              "params"], value_vars=[("net", "k_max")], value_name='k_max')
df2['W1'] = df2[('params', 'W1')]
df2['W2'] = df2[('params', 'W2')]
fg = sns.FacetGrid(df2, col='meds', col_wrap=4, aspect=1.5)
fg.map_dataframe(draw_heatmap, "W1", "W2", 'k_max', cbar=True,
                 square=True, vmin=df2['k_max'].min(), vmax=df2['k_max'].max())
fg.fig.get_axes()[0].invert_yaxis()

# rewire_n
df2 = df.melt(id_vars=["meds"]+[x for x in df.columns if x[0] ==
              "params"], value_vars=[("net", "rewire_n")], value_name='rewire_n')
df2['W1'] = df2[('params', 'W1')]
df2['W2'] = df2[('params', 'W2')]
fg = sns.FacetGrid(df2, col='meds', col_wrap=4, aspect=1.5)
fg.map_dataframe(draw_heatmap, "W1", "W2", 'rewire_n', cbar=True,
                 square=True, vmin=df2['rewire_n'].min(), vmax=df2['rewire_n'].max())
fg.fig.get_axes()[0].invert_yaxis()

# %%
# interactive
w1s = pd.unique(df[("params", "W1")])
W1 = ipywidgets.Select(
    options=w1s,
    rows=len(w1s),
    description='W1:',
    disabled=False,
)

w2s = pd.unique(df[("params", "W2")])
W2 = ipywidgets.Select(
    options=w2s,
    rows=len(w2s),
    description='W2:',
    disabled=False,
)

medSets = pd.unique(df[("meds")])
medSet = ipywidgets.Select(
    options=medSets,
    rows=len(medSets),
    description='Med Sets:',
    disabled=False,
)


def plot_bar(df, w1, w2, medSet):
    return df[df[("ws") == (w1, w2)] & df[("meds") == medSet]].plot.bar()


ipywidgets.interact(plot_bar, df=df, W1=W1, W2=W2, medSet=medSet)

w1, w2 = 0.5, 1.0
medSet = ("NO_MED", "GOOD_MED_X", "BAD_MED_X")
df2 = df_mean.reset_index(col_level=1).melt(id_vars=[(
    "", "meds"), ("", "ws")], value_vars=[("agents", "coop_freq")], value_name='coop')
df_mean.loc[[(('NO_MED', 'BAD_MED_X', 'FAIR_MED_X'), (0.5, 0.5))]]

w1, w2 = 0.5, 1.0
medSet = ("NO_MED", "GOOD_MED_X", "BAD_MED_X")
df2 = df_mean.reset_index(col_level=1).melt(id_vars=[(
    "", "meds"), ("", "ws")], value_vars=[("agents", "coop_freq")], value_name='coop')
df_mean.loc[[(('NO_MED', 'BAD_MED_X', 'FAIR_MED_X'), (0.5, 0.5))]]


def plot_bar(w1, w2, medSet):
    n = 6
    size = 4
    fig, ax = plt.subplots(1, n, figsize=((n+1)*size, size))
    plt.suptitle(f"Mediators: {medSet}, w1={w1}, w2={w2}")
    df_mean.loc[[(medSet, (w1, w2))]]['med_freqs'].plot.bar(
        ax=ax[0], xlabel="mediators", ylim=(0, 1), xticks=[])
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


ipywidgets.interact(plot_bar,
                    w1=list(pd.unique(df_mean[("params", "W1")])),
                    w2=list(pd.unique(df_mean[("params", "W2")])),
                    medSet=list(pd.unique(df_mean.index.get_level_values(0)))
                    )
