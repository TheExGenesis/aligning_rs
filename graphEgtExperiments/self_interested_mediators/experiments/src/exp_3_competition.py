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
episode_n = 1000
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
    results = [makeEntry2(res) for res in wsMatrixSim(medSet=medSet, w1s=w1s, w2s=w2s,
                                                      episode_n=episode_n, ts=ts, saveHistory=False, save=False) for i in range(n_trials)]
    # save whole run
    experiment_name = makeCompetitionName(
        {"medSet": [int2MedName[med] for med in medSet]})
    saveRes(results, experiment_name, dir_path)


# %%
