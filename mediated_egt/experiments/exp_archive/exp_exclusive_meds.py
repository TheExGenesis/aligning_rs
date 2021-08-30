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


def saveCompetitionExperiment(N=_N, episode_n=_episode_n, W1=_W, W2=_W2, graph=None, ts=(_T, _S), medStrats=None, strats=None, beta=0.005, k=30, medSet=_medSet, history=None, saveHistory=False, dir_path="./data"):
    print(f"running experiment. medSet {medSet} ts {ts}")
    run = cy_runCompetitionExperiment(N=N, episode_n=episode_n, W1=W1, W2=W2,
                                      ts=ts, beta=beta, k=k, saveHistory=True, history=[], medSet=medSet)
    print(f"saving {medSet, N, episode_n, ts, beta, W1, W2, k}")
    saveRes(run, makeCompetitionName(
            {"medSet": medSet, "N": N, "episode_n": episode_n, "ts": ts, "beta": beta, "W1": W1, "W2": W2, "k": k}))
    return run


def wsMatrixSim(medSet=[0, 1, 2, 3, 4], w1s=[0.5, 1, 2, 3], w2s=[0.5, 1, 2, 3], episode_n=10000, ts=(2.0, -1.0), saveHistory=False, save=False):
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
episode_n = 100
w1s = [0.5, 1, 2, 3]
w2s = [0.5, 1, 2, 3]
# sets of 2 mediators
medSets = [[medName2Int["NO_MED"], m] for m in exclusive] + [[medName2Int["RANDOM_MED"], m]
                                                             for m in exclusive] + [[medName2Int["FAIR_MED_X"], m] for m in [medName2Int["GOOD_MED_X"], medName2Int["BAD_MED_X"]]]
ts = (2.0, -1.0)


experiment_name = makeCompetitionName(
    {"n_eps": episode_n, "n_trials": n_trials})
# plt.suptitle(f"Avg. final med population and coop  {experiment_name}")

param_name = makeCompetitionName(
    {"medSet": "pairs", "ws1_ws2": w1s, "n_eps": episode_n, "n_trials": n_trials, "ts": ts})
run_name = f"exclusive_med_competition_{timestamp()}"
dir_path = f"../data/{run_name}"


size = 4
n = 2  # 2 plots per run, coop and med population
fig, ax = plt.subplots(len(medSets), n, figsize=(
    (n+1)*size, (len(medSets)+1)*size))
plt.suptitle(f"Competition pairs of mediators: {param_name}")

# run for each mediator
for i, medSet in enumerate(medSets):
    # run n_trials of matrix of games for each w
    results = [wsMatrixSim(medSet=medSet, w1s=w1s, w2s=w2s, episode_n=episode_n, ts=ts, saveHistory=False, save=False)
               for i in range(n_trials)]
    # save whole run
    experiment_name = makeCompetitionName(
        {"medSet": [int2MedName[med] for med in medSet]})
    saveRes(results, experiment_name, dir_path)
    coop_df = pd.DataFrame({k: [v] for k, v in coop(
        results).items()}).transpose().unstack()
    med_count_df = medCountsDf(results).transpose().unstack()
    plt.subplot(len(medSets), n, i*n+1)
    sns.heatmap(med_count_df[medSet[1]], annot=True, cbar=True, xticklabels=2,
                yticklabels=2, vmin=0, vmax=1, ax=plt.gca()).set(title=f"Avg. final counts of {int2MedName[medSet[1]]}, medSet:{medSet}")
    plt.subplot(len(medSets), n, i*n+2)
    sns.heatmap(coop_df, annot=True, cbar=True, xticklabels=4,
                yticklabels=2, vmin=0, vmax=1, ax=plt.gca()).set(title=f"Avg. final coop, medSet:{medSet}")
fig.savefig(f'../data/{run_name}/{param_name}.png')


# %%
