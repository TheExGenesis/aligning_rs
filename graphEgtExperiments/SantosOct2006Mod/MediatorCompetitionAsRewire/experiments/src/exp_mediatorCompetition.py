# %%
from mediators import NO_MED
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


def wsMatrixSim(medSet=[0, 1, 2, 3, 4], w1s=[0.5, 1, 2, 3], w2s=[0.5, 1, 2, 3], episode_n=10000, ts=(2.0, -1.0), saveHistory=False, save=True):
    N = 500
    beta = 0.005
    k = 30
    runs = {(w1, w2): cy_runCompetitionExperiment(N=N, episode_n=episode_n, W1=w1, W2=w2, ts=ts,
                                                  beta=beta, k=k, saveHistory=saveHistory, history=[], medSet=medSet) for w1, w2 in product(w1s, w2s)}
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
# sets of 2 mediators
medSets = [[medName2Int[NO_MED], medName2Int[m]] for m in [GOOD_MED, BAD_MED, FAIR_MED, RANDOM_MED]] + [[medName2Int[RANDOM_MED],
                                                                                                         medName2Int[m]] for m in [GOOD_MED, BAD_MED, FAIR_MED]] + [[medName2Int[FAIR_MED], medName2Int[m]] for m in [GOOD_MED, BAD_MED]]
ts = (2.0, -1.0)
# run for each mediator
for i, medSet in enumerate(medSets):
    # run n_trials of matrix of games for each w
    results = [wsMatrixSim(medSet=medSet, w1s=w1s, w2s=w2s, episode_n=episode_n, ts=ts, saveHistory=False, save=True)
               for i in range(n_trials)]
    # save whole run
    experiment_name = makeCompetitionName(
        {"medSet": [int2MedName[med] for med in medSet], "n_eps": episode_n, "n_trials": n_trials})+"_"+timestamp()
    saveRes(results, experiment_name,
            dir_path=f"../data/med_competition/{experiment_name}")
    c = coop(results)
    coop_df = pd.DataFrame(zip(list(c.values()), *transposeList(list(c.keys()))),
                           columns=["coops", 'w1s', 'w2s']).pivot('w2s', 'w1s', "coops").iloc[::-1]
    med_count_df = medCountsDf(results)
    print(med_count_df)
    plt.figure(i)  # creates a new figure
    sns.heatmap(coop_df, annot=True, cbar=True, xticklabels=2,
                yticklabels=2, vmin=0, vmax=1).set(title=f"Avg. final coop, {[int2MedName[med] for med in medSet]}, ep_n={episode_n}, n_trials={n_trials}")
    plt.savefig(
        f'../plots/med_competition/{experiment_name}.png')

# %%
