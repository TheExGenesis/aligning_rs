# %%
import matplotlib.pyplot as plt
from itertools import chain
from utils import transposeList
from plots import *
from egt_io import *
from defaultParams import _episode_n, _N, _W, _W2, _k, _T, _S
from defaultParams import *
from mediators2 import *
from mediators2 import _medSet
from evolution2 import *
from itertools import product, combinations
from functools import reduce
from dataframe import makeEntry2, makeColumns
import seaborn as sns
from math import inf


def wsMatrixSim(medSet=[0, 1, 2, 3, 4], w1s=[0.5, 1, 2, 3], w2s=[0.5, 1, 2, 3], episode_n=10000, ts=(2.0, -1.0), saveHistory=False, save=False):
    N = 500
    beta = 0.005
    k = 30
    # smallMedInit = True # if true, the mediator will be initialized with 90% no_med and 10% from meds in the set
    runs = [cy_runCompetitionExperiment(N=N, episode_n=episode_n, W1=w1, W2=w2, ts=ts,
                                        beta=beta, k=k, smallMedInit=True, saveHistory=saveHistory, history=[], medSet=medSet) for w1, w2 in product(w1s, w2s)]
    if save:
        print(f"saving {medSet, N, episode_n, beta, w1s, w2s, k}")
        saveRes(runs,   makeCompetitionName(
            {"medSet": medSet, "N": N, "episode_n": episode_n, "beta": beta, "W1": w1s, "W2": w2s, "k": k}),
            dir_path="../data")
    return runs

# single ts heatmap simulation


def tsMatrixSim(med=0, M=6, episode_n=10000, W1=1, saveHistory=False, save=False):
    gameParams = genTSParams(M)
    medSet = [med]
    N = 500
    W2 = 0
    beta = 0.005
    k = 30
    # runs = {ts: saveCompetitionExperiment(N =N, episode_n=episode_n, W1=W1, W2=W2, ts=ts, beta=beta, k=k, saveHistory=False, history=[], medSet=medSet) for ts in gameParams}
    runs = [runCompetitionExperiment(N=N, episode_n=episode_n, W1=W1, W2=W2, ts=ts,
                                     beta=beta, k=k, saveHistory=saveHistory, history=[], medSet=medSet) for ts in gameParams]
    if save:
        print(f"saving {medSet, N, episode_n, beta, W1, W2, k}")
        saveRes(runs,   makeCompetitionName(
            {"medSet": medSet, "N": N, "episode_n": episode_n, "beta": beta, "W1": W1, "W2": W2, "k": k}),
            dir_path="../data")
    # print("ending tsMatrixSim")
    return runs


# %%
% % time

# single meds
episode_n = 10
w1s = [0.5, 1, 2, 3, 4, inf]
# n_trials = 15
n_trials = 1
run_name = f"single_meds_ts_{timestamp()}"
dir_path = f"../data/{run_name}"
M = 7
M = 1
print(f"Running {run_name}")
for med in non_exclusive:
    ts_res = [tsMatrixSim(med=med, M=M, episode_n=episode_n, W1=w, save=False)
              for w in w1s for i in range(n_trials)]
    results = pd.DataFrame(
        [makeEntry2(res) for trial in ts_res for res in trial], columns=makeColumns()).fillna(0)
    experiment_name = makeCompetitionName({"med": int2MedName[med]})
    saveDf(results, experiment_name, dir_path)
print(f"Saved {run_name}")

# %%
% % time
# no rewire
M = 7
episode_n = 1000000
n_trials = 30
run_name = f"baseline_no_rewire_{timestamp()}"
dir_path = f"../data/{run_name}"
experiment_name = makeCompetitionName({"baseline": "no_rewire"})
print(f"Running {run_name}")
ts_res = [tsMatrixSim(
    med=0, M=M, episode_n=episode_n, W1=0, save=False) for i in range(n_trials)]
results = pd.DataFrame([makeEntry2(
    res) for trial in ts_res for res in trial], columns=makeColumns()).fillna(0)
saveDf(results, experiment_name, dir_path)
print(f"Saved {run_name}")

# %%
