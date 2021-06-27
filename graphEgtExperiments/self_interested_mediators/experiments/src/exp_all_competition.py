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
from dataframe import makeEntry2, makeColumns
import seaborn as sns
from math import inf


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
    return runs


# finds the W1 parameter in a filename in the form "...W1-{w}..."

# %%
# RUN N_TRIALS OF A MATRIX FOR EACH W AND \FOR EACH MEDIATOR


# run for each w and med


# %%
episode_n = 1000000
w1s = [0.5, 1, 2, 3, inf]
w2s = [0.5, 1, 2, 3]
ts = (2.0, -1.0)

# non-exclusive meds
n_trials = 100
run_name = f"vanilla_all_med_competition_{timestamp()}"
dir_path = f"../data/{run_name}"
experiment_name = makeCompetitionName({"medSet": "non_exclusive"})
print(f"Running {run_name}")
trials_results = [[makeEntry2(res) for res in wsMatrixSim(medSet=non_exclusive, w1s=w1s, w2s=w2s,
                                                          episode_n=episode_n, ts=ts, saveHistory=False, save=False)] for i in range(n_trials)]
results = pd.DataFrame(
    [r for res in trials_results for r in res], columns=makeColumns()).fillna(0)
saveDf(results, experiment_name, dir_path)
print(f"Saved {run_name}")

# exclusive meds
n_trials = 100
run_name = f"exclusive_all_med_competition_{timestamp()}"
dir_path = f"../data/{run_name}"
experiment_name = makeCompetitionName({"medSet": "exclusive"})
print(f"Running {run_name}")
trials_results = [[makeEntry2(res) for res in wsMatrixSim(medSet=exclusive, w1s=w1s, w2s=w2s,
                                                          episode_n=episode_n, ts=ts, saveHistory=False, save=False)] for i in range(n_trials)]
results = pd.DataFrame(
    [r for res in trials_results for r in res], columns=makeColumns()).fillna(0)
saveDf(results, experiment_name, dir_path)
print(f"Saved {run_name}")

# local-sensitive meds
n_trials = 20
run_name = f"local_all_med_competition_{timestamp()}"
dir_path = f"../data/{run_name}"
experiment_name = makeCompetitionName({"medSet": "local_meds"})
print(f"Running {run_name}")
trials_results = [[makeEntry2(res) for res in wsMatrixSim(medSet=local_meds, w1s=w1s, w2s=w2s,
                                                          episode_n=episode_n, ts=ts, saveHistory=False, save=False)] for i in range(n_trials)]
results = pd.DataFrame(
    [r for res in trials_results for r in res], columns=makeColumns()).fillna(0)
saveDf(results, experiment_name, dir_path)
print(f"Saved {run_name}")


# single meds
n_trials = 20
run_name = f"single_med_{timestamp()}"
dir_path = f"../data/{run_name}"
print(f"Running {run_name}")
for i, medSet in enumerate([[m] for m in non_exclusive]):
    # run n_trials of matrix of games for each w
    trials_results = [[makeEntry2(res) for res in wsMatrixSim(
        medSet=medSet, w1s=w1s, w2s=[0], episode_n=episode_n, ts=ts, saveHistory=False, save=False)] for i in range(n_trials)]
    results = pd.DataFrame(
        [r for res in trials_results for r in res], columns=makeColumns()).fillna(0)
    # save whole run
    experiment_name = makeCompetitionName(
        {"medSet": medSet})
    saveDf(results, experiment_name, dir_path)
print(f"Saved {run_name}")
, <
