from utils import transposeList
from mediators import _medSet
from mediators import int2MedName, medName2Int
from games import calcPayoffs, makeTSDilemma
from egt_io import *
from analysis import *
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
import numpy as np


def makeColumns():
    medNames = list(int2MedName.values())
    meds = list(product(["med"], int2MedName.values()))
    params = list(product(["params"], ["W1", "W2", "N", "episode_n"]))
    game = list(product(["game"], ["t", "s"]))
    med_freqs = list(product(["med_freqs"], int2MedName.values()))
    agent_stats = list(
        product(["agents"], ["coop_freq", "payoff_mean", "payoff_var"]))
    net_metrics = list(
        product(["net"], ["heterogeneity", "k_max", "rewire_n", "stop_n"]))
    meta_metrics = list(
        product(["meta"], ["timestamp"]))
    cols = pd.MultiIndex.from_tuples(
        meds+params+game+med_freqs+agent_stats+net_metrics+meta_metrics)
    return cols


def makeExperimentsDf(results):
    return pd.DataFrame(results, columns=makeColumns()).fillna(0)


# from most recent results which include parmams dict and rewire_n


def makeEntry2(res):
    params = {("params", "W1"): res["params"]["W1"], ("params", "W2"): res["params"]["W2"],
              ("params", "N"): res["params"]["N"], ("params", "episode_n"): res["params"]["episode_n"],
              ("params", "k"): res["params"]["k"], ("params", "beta"): res["params"]["beta"]}
    med = {("med", int2MedName[med]): 1 for med in res["params"]["medSet"]}
    game = {("game", "t"): res["params"]["t"],
            ("game", "s"): res["params"]["s"], }
    med_freqs = {("med_freqs", int2MedName[med]): cnt /
                 res["params"]["N"] for med, cnt in Counter(res["medStrats"]).items()}
    payoffs = np.array(calcPayoffs(makeTSDilemma(
        res["params"]["t"], res["params"]["s"]), res["graph"], res["finalStrats"]))
    agent_stats = {("agents", "coop_freq"): res["finalStrats"].mean(
    ), ("agents", "payoff_mean"): payoffs.mean(), ("agents", "payoff_var"): payoffs.var()}
    net_metrics = {("net", "heterogeneity"): heterogeneity(
        res['graph']), ("net", "k_max"): maxDegree(res['graph']), ("net", "rewire_n"): res['rewire_n'], ("net", "stop_n"): res['stop_n']}  # TODO
    net_metrics = {("meta", "timestamp"): res['timestamp']}
    df = {**params, **med, **game, **med_freqs, **agent_stats, **net_metrics}
    return df


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
    episode_n = 1000000
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

# load many different runs (as trials) rather than only loading trials if they're saved in the same file


def makeDataframeFromCompetitionRuns(results):
    data = []
    episode_n = 1000000
    N = 500
    t, s = 2, -1
    for medSet, runs in results.items():
        for trials in runs:
            for trial in trials:
                for (W1, W2), res in trial.items():
                    data.append(
                        makeEntry(res, N, episode_n, W1, W2, medSet, t, s))
    df = pd.DataFrame(data, columns=makeColumns()).fillna(0)
    return df


def example_load_data(filename="/mnt/c/Users/frsc/Documents/Projects/aligning_rs/graphEgtExperiments/SantosOct2006Mod/MediatorCompetitionAsRewire/experiments/data/med_competition/overnight_medSets-[most_pairs]_n_eps-1000000_n_trials-10_w1s-w2s-0.5-3_Jun-11-2021_1054.pkl"):
    return pickle.load(open(filename, "rb"))
