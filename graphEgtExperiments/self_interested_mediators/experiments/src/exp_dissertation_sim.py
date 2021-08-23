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
% % time
# W sweep of single meds
data_path = "jkkkkkkkkkkkkkkkkkkkk"
episode_n = 500000
n_trials = 15
total_res = []
for i in range(n_trials):
    for med in non_exclusive:
        ws = np.linspace(0.25, 10.25, 50)
        res = wSweepSim(ws, med=med, episode_n=episode_n)
        df_res = pd.DataFrame([makeEntry2(r)
                               for r in res], columns=makeColumns()).fillna(0)
        # saveDf(df_res, f"w_sweep_{int2MedName[med]}", data_path)
        total_res.append(df_res)
# concat all results in same df
df = pd.concat(total_res, ignore_index=True)
df = df.round(3)
df['meds'] = df['med'].apply(lambda r: tuple(
    c for c in df['med'].columns if r[c] == 1), axis=1)  # tuples of mediators
# tuples of ws pairs
df['ws'] = list(zip(df[('params', 'W1')], df[('params', 'W2')]))
df['ts'] = list(zip(df[('game', 't')], df[('game', 's')]))  # tuples t,s
saveDf(df, f"w_sweep", data_path)

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
df = pd.DataFrame([makeEntry2(
    res) for trial in ts_res for res in trial], columns=makeColumns()).fillna(0)

df = df.round(3)
df['meds'] = df['med'].apply(lambda r: tuple(
    c for c in df['med'].columns if r[c] == 1), axis=1)  # tuples of mediators
# tuples of ws pairs
df['ws'] = list(zip(df[('params', 'W1')], df[('params', 'W2')]))
df['ts'] = list(zip(df[('game', 't')], df[('game', 's')]))  # tuples t,s
saveDf(df, experiment_name, dir_path)
print(f"Saved {run_name}")

# %%
% % time
# NO_MED ts run
run_name = "sanity_check_nomed_ts"
dir_path = f"../data/{run_name}"
data_path = "/mnt/c/Users/frsc/Documents/Projects/aligning_rs/graphEgtExperiments/self_interested_mediators/experiments/data/aug23_singlemed_ts_run/"
episode_n = 500000
total_res = []
med = medName2Int["FAIR_MED_X"]
ws = [1]
n_trials = 10
runs = [tsMatrixSim(med=med, M=5, episode_n=episode_n, W1=w)
        for w in ws for i in range(n_trials)]
results = pd.DataFrame(
    [makeEntry2(res) for run in runs for res in run], columns=makeColumns()).fillna(0)
saveDf(results, "sanity_check_nomed_ts", dir_path)
# %%
# plot ts
df = results
df['meds'] = df['med'].apply(lambda r: tuple(
    c for c in df['med'].columns if r[c] == 1), axis=1)  # tuples of mediators
# tuples of ws pairs
df['ws'] = list(zip(df[('params', 'W1')], df[('params', 'W2')]))
df['ts'] = list(zip(df[('game', 't')], df[('game', 's')]))
filter_inf = True
df = df if not filter_inf else df[df[("params", "W1")] != inf]
df_mean_ts = df.groupby(["meds", ("params", "W1"), "ts"]).mean()
w1s = ws
medSet = ('FAIR_MED_X',)
w1 = 1
interpolate = False
df_pivoted = df_mean_ts.loc[medSet].loc[w1].pivot(
    index=[("game", "s")], columns=[("game", "t")]).sort_index(ascending=False)
ss, ts = ['%.2f' % elem for elem in df_pivoted[("agents", "coop_freq")].index], [
    '%.2f' % elem for elem in df_pivoted[("agents", "coop_freq")].columns]
plot_heatmap(df_pivoted[("agents", "coop_freq")], vmin=0, vmax=1, yticklabels=ss,
             xticklabels=ts, title=f"Avg. final #cooperators", interpolate=interpolate)


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
# n_trials = 15
n_trials = 1

# exclusive meds
run_name = f"w2_sweep_all_med_competition_smallinit_{timestamp()}"
dir_path = f"../data/{run_name}"
experiment_name = makeCompetitionName({"medSet": "exclusive"})
# smallMedInit = True # if true, the mediator will be initialized with 90% no_med and 10% from meds in the set
medSet = exclusive[1:]  # exclude no_med
# %%
% % time
runs = [runCompetitionExperiment(N=N, episode_n=episode_n, W1=w1, W2=w2, ts=ts,
                                 beta=beta, k=k, saveHistory=False, history=[], medSet=medSet, smallMedInit=True) for w1, w2 in product(w1s, w2s) for i in range(n_trials)]

print(f"Running {run_name}")
results = pd.DataFrame(
    [makeEntry2(res) for res in runs], columns=makeColumns()).fillna(0)
saveDf(results, experiment_name, dir_path)
print(f"Saved {run_name}")

# %%
