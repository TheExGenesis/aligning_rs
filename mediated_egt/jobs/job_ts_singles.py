# %%
from egt_mediators.plots import plot_heatmap
from egt_mediators.numba.evolution import *
import matplotlib.pyplot as plt
from itertools import chain
from egt_mediators.utils import transposeList
from egt_mediators.plots import *
from egt_mediators.egt_io import *
from egt_mediators.defaultParams import _episode_n, _N, _W, _W2, _k, _T, _S
from egt_mediators.defaultParams import *
from egt_mediators.numba.mediators import *
from egt_mediators.numba.mediators import _medSet
from itertools import product, combinations
from functools import reduce
from egt_mediators.dataframe import makeEntry2, makeColumns
import seaborn as sns
from math import inf


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
# % % time

# single meds
episode_n = 1000000
w1s = [0.5, 1, 2, 3, 4, inf]
n_trials = 10
run_name = f"single_meds_ts_{timestamp()}"
dir_path = f"../data/{run_name}"
M = 7
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
