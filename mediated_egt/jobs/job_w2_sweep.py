# %%
from plots import plot_heatmap
from evolution2 import *
import matplotlib.pyplot as plt
from itertools import chain
from utils import transposeList
from plots import *
from egt_io import *
from defaultParams import _episode_n, _N, _W, _W2, _k, _T, _S
from defaultParams import *
from mediators2 import *
from mediators2 import _medSet
from itertools import product, combinations
from functools import reduce
from dataframe import makeEntry2, makeColumns
import seaborn as sns
from math import inf
# %%


def wSweepSim(ws, med=0, episode_n=500000, saveHistory=False, save=True):
    medSet = [med]
    N = 500
    W2 = 0
    beta = 0.005
    k = 30
    ts = (2, -1)
    # runs = {ts: saveCompetitionExperiment(N =N, episode_n=episode_n, W1=W1, W2=W2, ts=ts, beta=beta, k=k, saveHistory=False, history=[], medSet=medSet) for ts in gameParams}
    runs = [runCompetitionExperiment(N=N, episode_n=episode_n, W1=w, W2=W2, ts=ts,
                                     beta=beta, k=k, saveHistory=saveHistory, history=[], medSet=medSet) for w in ws]
    print("ending wSweepSim")
    return runs


def tsMatrixSim(med=0, M=6, episode_n=10000, W1=1, saveHistory=False, save=True):
    gameParams = genTSParams(M)
    medSet = [med]
    N = 500
    W2 = 0
    beta = 0.005
    k = 30
    # runs = {ts: saveCompetitionExperiment(N =N, episode_n=episode_n, W1=W1, W2=W2, ts=ts, beta=beta, k=k, saveHistory=False, history=[], medSet=medSet) for ts in gameParams}
    runs = [runCompetitionExperiment(N=N, episode_n=episode_n, W1=W1, W2=W2, ts=ts,
                                     beta=beta, k=k, saveHistory=saveHistory, history=[], medSet=medSet) for ts in gameParams]
    print("ending tsMatrixSim")
    return runs


# %%
# Competition between all mediators with fixed W1 = 1, varying W2 from 0 to 0.1 with step 0.01.
episode_n = 10000000
# episode_n = 1000
w1s = [1]
w2s = np.linspace(0, 0.2, 20)
ts = (2.0, -1.0)
N = 1000
beta = 0.005
k = 30
n_trials = 15
# n_trials = 1

# exclusive meds
run_name = f"w2_sweep_all_med_competition_smallinit_{timestamp()}"
dir_path = f"../data/{run_name}"
experiment_name = makeCompetitionName({"medSet": "exclusive"})
# smallMedInit = True # if true, the mediator will be initialized with 90% no_med and 10% from meds in the set
medSet = exclusive[1:]  # exclude no_med
# %%
# %%time
runs = [runCompetitionExperiment(N=N, episode_n=episode_n, W1=w1, W2=w2, ts=ts,
                                 beta=beta, k=k, saveHistory=False, history=[], medSet=medSet, smallMedInit=True) for w1, w2 in product(w1s, w2s) for i in range(n_trials)]

print(f"Running {run_name}")
results = pd.DataFrame(
    [makeEntry2(res) for res in runs], columns=makeColumns()).fillna(0)
saveDf(results, experiment_name, dir_path)
print(f"Saved {run_name}")

# %%
