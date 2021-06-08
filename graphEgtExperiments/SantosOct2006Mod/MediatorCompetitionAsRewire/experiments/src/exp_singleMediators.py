# %%
import matplotlib.pyplot as plt
from itertools import chain
from utils import transposeList
from plots import *
from egt_io import *
from defaultParams import _episode_n, _N, _W, _W2, _k, _T, _S
from defaultParams import *
from mediators import _medSet
from evolution import *


def cy_competitionFullEvo(episode_n=1000, medSet=[0, 1], saveHistory=False, save=False):
    n_trials = 1
    N = 500
    W1 = 1
    W2 = 1
    ts = (2, -1)
    beta = 0.005
    k = 30
    M = 5
    run = cy_runCompetitionExperiment(N=N, episode_n=episode_n, W1=W1, W2=W2, ts=ts,
                                      beta=beta, k=k, saveHistory=False, history=[], medSet=medSet)
    return run


def saveCompetitionExper    iment(N=_N, episode_n=_episode_n, W1=_W, W2=_W2, graph=None, ts=(_T, _S), medStrats=None, strats=None, beta=0.005, k=30, medSet=_medSet, history=None, saveHistory=False, dir_path="./data"):
    print(f"running experiment. medSet {medSet} ts {ts}")
    run = cy_runCompetitionExperiment(N=N, episode_n=episode_n, W1=W1, W2=W2,
                                      ts=ts, beta=beta, k=k, saveHistory=True, history=[], medSet=medSet)
    print(f"saving {medSet, N, episode_n, ts, beta, W1, W2, k}")
    saveRes(run, makeCompetitionName(
            {"medSet": medSet, "N": N, "episode_n": episode_n, "ts": ts, "beta": beta, "W1": W1, "W2": W2, "k": k}))
    return run


def tsMatrixSim(med=0, M=2, episode_n=10000, W1=1, saveHistory=False, save=True):
    gameParams = genTSParams(M)
    medSet = [med]
    N = 500
    W2 = 0
    beta = 0.005
    k = 30
    # runs = {ts: saveCompetitionExperiment(N =N, episode_n=episode_n, W1=W1, W2=W2, ts=ts, beta=beta, k=k, saveHistory=False, history=[], medSet=medSet) for ts in gameParams}
    runs = {ts: cy_runCompetitionExperiment(N=N, episode_n=episode_n, W1=W1, W2=W2, ts=ts,
                                            beta=beta, k=k, saveHistory=saveHistory, history=[], medSet=medSet) for ts in gameParams}
    if save:
        print(f"saving {medSet, N, episode_n, beta, W1, W2, k}")
        saveRes(runs,   makeCompetitionName(
            {"medSet": medSet, "N": N, "episode_n": episode_n, "beta": beta, "W1": W1, "W2": W2, "k": k}),
            dir_path="../data")
    print("ending tsMatrixSim")
    return runs

# list of dicts


def list2Dict(runs):
    all_keys = set(chain(*[x.keys() for x in runs]))
    return {k: [run[k] for run in runs] for k in all_keys}

# dict of lists


def coop(runs):
    runsByGame = list2Dict(runs).items()
    N = list(runs[0].values())[0]["initStrats"].size
    return {k: np.array([run['finalStrats'] for run in gameRuns]).sum(axis=1).mean()/N for k, gameRuns in runsByGame}


def plotCoop(results):
    c = coop(results)
    df = pd.DataFrame(zip(list(c.values()), *transposeList(list(c.keys()))),
                      columns=["coops", 't', 's']).pivot('s', 't', "coops").iloc[::-1]
    fig = sns.heatmap(df, annot=True, cbar=True, xticklabels=2,
                      yticklabels=2, vmin=0, vmax=1).set(title="Avg. final cooperators")


# finds the W1 parameter in a filename in the form "...W1-{w}..."


def filename2W(fn):
    import re
    return int(re.search(r".*W1-([0-9]+).*", fn).group(1))


# loads a pickle
def loadPickle(path):
    import pickle
    with open(path, "rb") as file:
        res = pickle.load(file)
        print(f"loaded {path}")
        return res

#  dict comprehension where values are lists keys are given by applying a function fn to items


def dictOfLists(my_list, fn=lambda x: x):
    new_dict = {}
    for value in my_list:
        key = fn(value)
        if key in new_dict:
            new_dict[key].append(value)
        else:
            new_dict[key] = [value]
    return new_dict

# recursively find all pickles in a dir


def getAllPickles(dir_name):
    return [os.path.join(root, name)
            for root, dirs, files in os.walk("../data")
            for name in files
            if name.endswith(".pkl")]
# load all filenames in dir and its subdirs into 1 experiment dict where keys are W values


def loadExperiment(dir_name):
    filenames = getAllPickles(dir_name)
    res = {k: [loadPickle(fn) for fn in fns]
           for k, fns in dictOfLists(filenames, filename2W).items()}
    return res


# n_trials = 1
# # results = [tsMatrixSim(M=5, episode_n=10000, save=True) for i in range(n_trials)]
# results = [tsMatrixSim(M=5, episode_n=10000, save=False)
#            for i in range(n_trials)]
# plotCoop(results)
# fig.savefig("./data/coopLandscape.png")

# %%
% % time
# run for each w and plot
n_trials = 10
episode_n = 100000
ws = [0, 1, 2, 3, 4]
# run n_trials of matrix of games for each w
w_results = {w: [tsMatrixSim(M=5, episode_n=episode_n, W1=w, save=True)
                 for i in range(n_trials)] for w in ws}
# save all of them
saveRes(runs,   makeCompetitionName({"episode_n": episode_n, "ws": ws, "n_trials": n_trials}),
        dir_path=f"../data/exp_no_med_{timestamp()}")
size = 4
n = len(list(w_results.items()))
fig, ax = plt.subplots(1, n, figsize=((n+1)*size, size))
for i, (k, results) in enumerate(w_results.items()):
    c = coop(results)
    df = pd.DataFrame(zip(list(c.values()), *transposeList(list(c.keys()))),
                      columns=["coops", 't', 's']).pivot('s', 't', "coops").iloc[::-1]
    plt.subplot(1, n, i+1)
    sns.heatmap(df, annot=True, cbar=True, xticklabels=2,
                yticklabels=2, vmin=0, vmax=1, ax=plt.gca()).set(title=f"Avg. final cooperators, W1={k}, ep_n={episode_n}")
    fig.savefig(
        f'../plots/{makeCompetitionName({"medSet":"NO_MED", "n_eps":episode_n, "n_trials": n_trials})}.png')

# %%


# %%
# CONTINUE RUN, SAVE, PLOT, SAVE PLOT

# load run
w_results = loadExperiment("../data/NO_MED-100k")
# continue run
ws = [0, 1, 2, 3, 4]
n_trials = 10
gameParams = genTSParams(5)
for w in ws:
    for i in range(n_trials):
        for ts in gameParams:
            cur_res = w_results[w][i][ts]
            w_results[w][i][ts] = cy_continueCompetitionExperiment(
                cur_res['graph'], cur_res['medStrats'], cur_res['finalStrats'], N=500, episode_n=100000, W1=w, W2=0, ts=ts, beta=0.005, k=30, medSet=[0])
# save whole run
episode_n = 100000
experiment_name = makeCompetitionName(
    {"medSet": "NO_MED", "n_eps": episode_n*3, "n_trials": n_trials})+"_"+timestamp()
saveRes(w_results, experiment_name,
        dir_path=f"../data/{experiment_name}")
size = 4
n = len(list(w_results.items()))
fig, ax = plt.subplots(1, n, figsize=((n+1)*size, size))
for i, (k, results) in enumerate(w_results.items()):
    c = coop(results)
    df = pd.DataFrame(zip(list(c.values()), *transposeList(list(c.keys()))),
                      columns=["coops", 't', 's']).pivot('s', 't', "coops").iloc[::-1]
    plt.subplot(1, n, i+1)
    sns.heatmap(df, annot=True, cbar=True, xticklabels=2,
                yticklabels=2, vmin=0, vmax=1, ax=plt.gca()).set(title=f"Avg. final cooperators, W1={k}, ep_n={episode_n}")
    fig.savefig(
        f'../plots/{experiment_name}.png')


# %%
# RUN N_TRIALS OF A MATRIX FOR EACH W AND FOR EACH MEDIATOR

# run for each w and med
n_trials = 10
episode_n = 100000
ws = [0.5, 1, 2, 3]
medSet = [1, 2, 3, 4]
# run for each mediator
for med in medSet:
    # run n_trials of matrix of games for each w
    w_results = {w: [tsMatrixSim(med=med, M=5, episode_n=episode_n, W1=w, save=True)
                     for i in range(n_trials)] for w in ws}
    # save whole run
    experiment_name = makeCompetitionName(
        {"medSet": int2MedName[med], "n_eps": episode_n, "n_trials": n_trials})+"_"+timestamp()
    saveRes(w_results, experiment_name,
            dir_path=f"../data/{experiment_name}")
    size = 4
    n = len(list(w_results.items()))
    fig, ax = plt.subplots(1, n, figsize=((n+1)*size, size))
    for i, (k, results) in enumerate(w_results.items()):
        c = coop(results)
        df = pd.DataFrame(zip(list(c.values()), *transposeList(list(c.keys()))),
                          columns=["coops", 't', 's']).pivot('s', 't', "coops").iloc[::-1]
        plt.subplot(1, n, i+1)
        sns.heatmap(df, annot=True, cbar=True, xticklabels=2,
                    yticklabels=2, vmin=0, vmax=1, ax=plt.gca()).set(title=f"Avg. final coop, {int2MedName[med]}, W1={k}, ep_n={episode_n}, n_trials={n_trials}")
        fig.savefig(
            f'../plots/{experiment_name}.png')

# %%
