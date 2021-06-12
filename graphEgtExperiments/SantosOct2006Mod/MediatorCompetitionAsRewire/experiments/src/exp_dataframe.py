# make comprehensive dataframe from experiment results
# %%
from utils import transposeList
from mediators import int2MedName, medName2Int
from egt_io import *
from analysis import *
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product

# from overnight experiment of different pairs of mediators and differente combinations of w1 and w2


def makeColumns():
    medNames = list(int2MedName.values())
    meds = list(product(["med"], int2MedName.values()))
    params = list(product(["params"], ["W1", "W2", "N", "episode_n"]))
    game = list(product(["game"], ["t", "s"]))
    med_freqs = list(product(["med_freqs"], int2MedName.values()))
    agent_stats = list(product(["agents"], ["coop_freq"]))
    net_metrics = list(product(["net"], ["heterogeneity", "k_max"]))
    cols = pd.MultiIndex.from_tuples(
        meds+params+game+med_freqs+agent_stats+net_metrics)
    return cols


def makeDfMedWs(results):
    pd.DataFrame(columns=makeColumns())


def makeEntry(res, N, episode_n, W1, W2, medSet, t, s):
    params = {("params", "W1"): W1, ("params", "W2"): W2,
              ("params", "N"): N, ("params", "episode_n"): episode_n}
    med = {("med", int2MedName[med]): 1 for med in medSet}
    game = {("game", "t"): t, ("game", "s"): s, }
    med_freqs = {("med_freqs", int2MedName[med]): cnt /
                 N for med, cnt in Counter(res["medStrats"]).items()}
    agent_stats = {("agents", "coop_freq"): res["finalStrats"].sum()/N}
    net_metrics = {("net", "heterogeneity"): heterogeneity(
        res['graph']), ("net", "k_max"): maxDegree(res['graph'])}  # TODO
    df = {**params, **med, **game, **med_freqs, **agent_stats, **net_metrics}
    return df

# (meds->trials->ws->res)


def makeDataframeFromOvernightRuns(results):
    data = []
    episode_n = 1000000
    N = 500
    t, s = 2, -1
    for medSet, trials in results.items():
        for trial in trials:
            for (W1, W2), res in trial.items():
                data.append(makeEntry(res, N, episode_n, W1, W2, medSet, t, s))
    df = pd.DataFrame(data, columns=makeColumns()).fillna(0)
    return df

# (meds->files->w->trials->t,s->)


def makeDataframeFromSingleMedRuns(results):
    data = []
    episode_n = 100000
    N = 500
    for med, files in results.items():
        for file in files:
            for (W1, trials) in file.items():
                for trial in trials:
                    for (t, s), res in trial.items():
                        W2 = 0
                        data.append(makeEntry(res, N, episode_n,
                                    W1, W2, [medName2Int[med]], t, s))
    df = pd.DataFrame(data, columns=makeColumns()).fillna(0)
    return df


def example_load_data(filename="/mnt/c/Users/frsc/Documents/Projects/aligning_rs/graphEgtExperiments/SantosOct2006Mod/MediatorCompetitionAsRewire/experiments/data/med_competition/overnight_medSets-[most_pairs]_n_eps-1000000_n_trials-10_w1s-w2s-0.5-3_Jun-11-2021_1054.pkl"):
    return pickle.load(open(filename, "rb"))

# makeDataframeFromOvernightRuns(example_load_data())

# res = results[(0,1)][0][(1,1)]


# plot coop matrix
def plotCoopFromDf(df, med, W1, W2):
    df_games = df[(df["med", med] == 1) & (
        df["params", "W2"] == W2) & (df["params", "W1"] == W1)]
    df_select = df_games.loc[:, ["game", "agents"]]
    df_axed = df_select.groupby([("game", "s"), ("game", "t")]).mean()
    df_matrix = df_axed.unstack().sort_index(ascending=False)
    ax = sns.heatmap(df_matrix, annot=True, cbar=True, xticklabels=df_matrix.index,
                     yticklabels=df_matrix.index, vmin=0, vmax=1, ax=plt.gca()).set(xlabel='t', ylabel='s', title=f"{med}, W1={W1}")
    return ax

# for all w1, med combinations


def plotAllCoopsFromDf(df):
    size = 4
    n = len(_medSet)
    fig, ax = plt.subplots(1, n, figsize=((n+1)*size, size))
    for i, med in enumerate(_medSet):
        plt.subplot(1, n, i+1)
        axis = plotCoopFromDf(df, med, 1, 0)


# loads all pickles in the dir
results = loadExperiment("../data/single_med", filename2Med)
df = makeDataframeFromSingleMedRuns(results)
saveRes(df, "single_med_df", dir_path=f"../data/single_med/")
