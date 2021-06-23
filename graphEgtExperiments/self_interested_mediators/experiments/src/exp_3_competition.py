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


def wsMatrixSim(medSet=[0, 1, 2, 3, 4], w1s=[0.5, 1, 2, 3], w2s=[0.5, 1, 2, 3], episode_n=10000, ts=(2.0, -1.0), saveHistory=False, save=False):
    N = 500
    beta = 0.005
    k = 30
    runs = [cy_runCompetitionExperiment(N=N, episode_n=episode_n, W1=w1, W2=w2, ts=ts,
                                        beta=beta, k=k, saveHistory=saveHistory, history=[], medSet=medSet) for w1, w2 in product(w1s, w2s)]
    if save:
        print(f"saving {medSet, N, episode_n, beta, w1s, w2s, k}")
        saveRes(runs,   makeCompetitionName(
            {"medSet": medSet, "N": N, "episode_n": episode_n, "beta": beta, "W1": w1s, "W2": w2s, "k": k}),
            dir_path="../data")
    print("ending wsMatrixSim")
    return runs


# finds the W1 parameter in a filename in the form "...W1-{w}..."

# %%
# RUN N_TRIALS OF A MATRIX FOR EACH W AND \FOR EACH MEDIATOR


# run for each w and med
n_trials = 10
episode_n = 1000000
w1s = [0.5, 1, 2, 3]
w2s = [0.5, 1, 2, 3]
# sets of 3 mediators
medSets = [*[[medName2Int["NO_MED"], *advs] for advs in combinations(exclusive, 2)],
           *[[medName2Int["RANDOM_MED"], *advs] for advs in combinations(exclusive, 2)]]
ts = (2.0, -1.0)


experiment_name = makeCompetitionName(
    {"n_eps": episode_n, "n_trials": n_trials})
# plt.suptitle(f"Avg. final med population and coop  {experiment_name}")

param_name = makeCompetitionName(
    {"medSet": "pairs", "ws1_ws2": w1s, "n_eps": episode_n, "n_trials": n_trials, "ts": ts})
run_name = f"exclusive_3_med_competition_{timestamp()}"
dir_path = f"../data/{run_name}"

# %%
# run for each mediator
for i, medSet in enumerate(medSets):
    # run n_trials of matrix of games for each w
    trials_results = [[makeEntry2(res) for res in wsMatrixSim(
        medSet=medSet, w1s=w1s, w2s=w2s, episode_n=episode_n, ts=ts, saveHistory=False, save=False)] for i in range(n_trials)]
    results = [r for res in trials_results for r in res]
    # save whole run
    experiment_name = makeCompetitionName(
        {"medSet": [int2MedName[med] for med in medSet]})
    saveRes(results, experiment_name, dir_path)


# %%
# loading
def plotting_df():
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
    df.groupby(['ws', 'meds']).mean()
    # barplot grid
    df_freqs = df_means['med_freqs'].reset_index()
    tidy = df_freqs.melt(id_vars=['ws', 'meds'], value_vars=df['med'].columns)
    sns.catplot(x='meds', y='value', hue='variable', col="ws",
                data=tidy, kind="bar", col_wrap=4, aspect=4)

# %%
# heatmap for coop/heterogeneity/etc


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
