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


def saveCompetitionExperiment(N=_N, episode_n=_episode_n, W1=_W, W2=_W2, graph=None, ts=(_T, _S), medStrats=None, strats=None, beta=0.005, k=30, medSet=_medSet, history=None, saveHistory=False, dir_path="./data"):
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


# %%
# RUN N_TRIALS OF A MATRIX FOR EACH W AND FOR EACH MEDIATOR
# run for each w and med
n_trials = 10
episode_n = 10
ws = [0.5, 1, 2, 3]
medSet = [1, 2, 3, 4]
# run for each mediator

size = 4
n = len(ws)
fig, ax = plt.subplots(len(medSet), n, figsize=(
    (n+1)*size, (len(medSet)+1)*size))
experiment_name = makeCompetitionName(
    {"n_eps": episode_n, "n_trials": n_trials})
plt.suptitle(f"Avg. final coop {experiment_name}")


run_name = f"single_med_{timestamp()}"
dir_path = f"../data/{run_name}"
for j, med in enumerate(medSet):
    # run n_trials of matrix of games for each w
    w_results = {w: [tsMatrixSim(med=med, M=5, episode_n=episode_n, W1=w, save=False)
                     for i in range(n_trials)] for w in ws}
    # save whole run
    experiment_name = makeCompetitionName(
        {"medSet": int2MedName[med], "n_eps": episode_n, "n_trials": n_trials})+"_"+timestamp()
    saveRes(w_results, experiment_name, dir_path=dir_path)
    for i, (k, results) in enumerate(w_results.items()):
        c = coop(results)
        df = pd.DataFrame(zip(list(c.values()), *transposeList(list(c.keys()))),
                          columns=["coops", 't', 's']).pivot('s', 't', "coops").iloc[::-1]
        plt.subplot(len(medSet), n, j*n+i+1)
        sns.heatmap(df, annot=True, cbar=True, xticklabels=2,
                    yticklabels=2, vmin=0, vmax=1, ax=plt.gca()).set(title=f"{int2MedName[med]}, W1={k}")
fig.savefig(
    f'../data/{run_name}/{experiment_name}.png')

# %%
