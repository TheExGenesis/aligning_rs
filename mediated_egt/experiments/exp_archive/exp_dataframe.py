# make comprehensive dataframe from experiment results
# %%
from utils import transposeList
from mediators import _medSet
from mediators import int2MedName, medName2Int
from egt_io import *
from analysis import *
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
from dataframe import *
# from overnight experiment of different pairs of mediators and differente combinations of w1 and w2


# makeDataframeFromOvernightRuns(example_load_data())

# res = results[(0,1)][0][(1,1)]


# plot coop matrix where axes are t,s
def plotCoopFromDf(df, med, W1, W2):
    df_games = df[(df["med", med] == 1) & (
        df["params", "W2"] == W2) & (df["params", "W1"] == W1)]
    df_select = df_games.loc[:, ["game", "agents"]]
    df_axed = df_select.groupby([("game", "s"), ("game", "t")]).mean()
    df_matrix = df_axed.unstack().sort_index(ascending=False)
    ax = sns.heatmap(df_matrix, annot=True, cbar=True, xticklabels=df_matrix.columns.get_level_values(2),
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

# plot coop matrix where axes are w1, w2. meds are string[]


def plotCoopWsFromDf(df, meds):
    df_games = df[(df["med", meds[0]] == 1) & (df["med", meds[1]] == 1)]
    df_select = df_games.loc[:, [
        ("params", "W1"), ("params", "W2"), ("agents", "coop_freq")]]
    df_axed = df_select.groupby([("params", "W1"), ("params", "W2")]).mean()
    df_matrix = df_axed.unstack().sort_index(ascending=False)
    ax = sns.heatmap(df_matrix, annot=True, cbar=True, xticklabels=df_matrix.columns.get_level_values(2),
                     yticklabels=df_matrix.index, vmin=0, vmax=1, ax=plt.gca()).set(xlabel='W1', ylabel='W2', title=f"\nAvg final coop rate in {meds}")
    return ax


def plotMedCountsWsFromDf(df, meds, idx=1):
    df_games = df[(df["med", meds[0]] == 1) & (df["med", meds[1]] == 1)]
    df_select = df_games.loc[:, [
        ("params", "W1"), ("params", "W2"), ("med_freqs", meds[idx])]]
    df_axed = df_select.groupby([("params", "W1"), ("params", "W2")]).mean()
    df_matrix = df_axed.unstack().sort_index(ascending=False)
    ax = sns.heatmap(df_matrix, annot=True, cbar=True, xticklabels=df_matrix.columns.get_level_values(2),
                     yticklabels=df_matrix.index, vmin=0, vmax=1, ax=plt.gca()).set(xlabel='W1', ylabel='W2', title=f"\n{meds[idx]} avg final counts in {meds}")
    return ax

# for each medSet, plot medCounts and coop


def plotAllMedCompetitionFromDf(df, medSets):
    size = 4
    n = len(medSets)
    m = 3  # coop, medcount1 and medcount2
    fig, ax = plt.subplots(n, m, figsize=((m+1.5)*size, (n+1)*size))
    for i, medSet in enumerate(medSets):
        plt.subplot(n, m, m*i+1)
        plotMedCountsWsFromDf(df, medSet, idx=1)
        plt.subplot(n, m, m*i+2)
        plotMedCountsWsFromDf(df, medSet, idx=0)
        plt.subplot(n, m, m*i+3)
        plotCoopWsFromDf(df, medSet)
    plt.tight_layout()
    return fig


# %%
# loads all pickles in the dir
experiment_name = "single_med_medSet-1-4_ws-0.5-3_n_eps-1000000_n_trials-10_jun-14-2021_1732"
results = loadExperiment(
    "../data/"+experiment_name, filename2Med)
df = makeDataframeFromSingleMedRuns(results)
saveRes(df, "df_"+experiment_name, dir_path=f"../data/"+experiment_name)

# %%
# exclusive mediator competition
# loads all pickles in the dir
experiment_name = "exclusive_med_competition_jun-14-2021_1858"
results = loadExperiment(
    "../data/"+experiment_name, filename2MedPair)
df = makeDataframeFromOvernightRuns(results)
saveRes(df, "df_"+experiment_name, dir_path=f"../data/"+experiment_name)

exclusive = [medName2Int[name]
             for name in ["GOOD_MED_X", "BAD_MED_X", "RANDOM_MED_X", "FAIR_MED_X"]]

medSets = [[medName2Int["NO_MED"], m] for m in exclusive] + [[medName2Int["RANDOM_MED"], m]
                                                             for m in exclusive] + [[medName2Int["FAIR_MED_X"], m] for m in [medName2Int["GOOD_MED_X"], medName2Int["BAD_MED_X"]]]
medSetNames = [[int2MedName[m] for m in medSet] for medSet in medSets]
fig = plotAllMedCompetitionFromDf(df, medSetNames)
fig.savefig("../data/"+experiment_name+".png")


# %%
# competition (not exclusive)
experiment_name = "med_competition"
results = loadPickle(
    "/mnt/c/Users/frsc/Documents/Projects/aligning_rs/graphEgtExperiments/SantosOct2006Mod/MediatorCompetitionAsRewire/experiments/data/med_competition/overnight_medSets-[most_pairs]_n_eps-1000000_n_trials-10_w1s-w2s-0.5-3_Jun-11-2021_1054.pkl")
df = makeDataframeFromOvernightRuns(results)
medSets = [[medName2Int["NO_MED"], medName2Int[m]] for m in ["GOOD_MED", "BAD_MED", "FAIR_MED", "RANDOM_MED"]] + [[medName2Int["RANDOM_MED"],
                                                                                                                   medName2Int[m]] for m in ["GOOD_MED", "BAD_MED", "FAIR_MED"]] + [[medName2Int["FAIR_MED"], medName2Int[m]] for m in ["GOOD_MED", "BAD_MED"]]
saveRes(df, "df_"+experiment_name, dir_path=f"../data/"+experiment_name)
medSetNames = [[int2MedName[m] for m in medSet] for medSet in medSets]
fig = plotAllMedCompetitionFromDf(df, medSetNames)
fig.savefig("../data/"+experiment_name+".png")

# %%
