# Ran TS with our numba optimized code
# results came back unexpected: coop_freqs that should be 0 on average are 0.5
# stop_n way 10x as high as expected
# So first thing I'm gonna do is reproduce the bug with just 1 med and 1 W and plot it here
# SUCCESS
# np.random.randint(N-1) instead of np.random.randint(N) was leaving 1 node with fixed strat forever, making a star around it
# %%
from egt_mediators.plots import *
# %%
import matplotlib.pyplot as plt
from itertools import chain
from egt_mediators.utils import transposeList
from egt_mediators.defaultParams import _episode_n, _N, _W, _W2, _k, _T, _S
from egt_mediators.defaultParams import *
from egt_mediators.egt_io import *
from egt_mediators.numba.mediators import *
from egt_mediators.numba.mediators import _medSet
from egt_mediators.numba.evolution import *
from itertools import product, combinations
from functools import reduce
from egt_mediators.dataframe import makeEntry2, makeColumns
import seaborn as sns
from math import inf

# single ts heatmap simulation


def tsMatrixSim(med=0, M=6, episode_n=10000, W1=1, saveHistory=False):
    gameParams = genTSParams(M)
    medSet = [med]
    N = 500
    W2 = 0
    beta = 0.005
    k = 30
    # runs = {ts: saveCompetitionExperiment(N =N, episode_n=episode_n, W1=W1, W2=W2, ts=ts, beta=beta, k=k, saveHistory=False, history=[], medSet=medSet) for ts in gameParams}
    runs = [runCompetitionExperiment(N=N, episode_n=episode_n, W1=W1, W2=W2, ts=ts,
                                     beta=beta, k=k, saveHistory=saveHistory, history=[], medSet=medSet) for ts in gameParams]
    return runs


# %%
# single meds
episode_n = 2000000
# episode_n = 10
# w1s = [0.5, 1, 2, 3, 4]
w1s = [0.5]
n_trials = 10
# n_trials = 1
run_name = f"single_meds_ts_{timestamp()}"
dir_path = f"../data/{run_name}"
M = 5
# M = 2
dfs = []
# mediators = non_exclusive
medSet = [0]

# run shit
# %%
% % time
print(f"Running {run_name}")
for med in medSet:
    ts_res = [tsMatrixSim(med=med, M=M, episode_n=episode_n, W1=w)
              for w in w1s for i in range(n_trials)]
    results = pd.DataFrame(
        [makeEntry2(res) for trial in ts_res for res in trial], columns=makeColumns()).fillna(0)
    experiment_name = makeCompetitionName({"med": int2MedName[med]})
    dfs.append(results)
    # saveDf(results, experiment_name, dir_path)
print(f"Saved {run_name}")

#
# %%
df = pd.concat(dfs)
df = df.round(3)
df = df.rename(columns={'Unnamed: 44_level_1': '',
               'Unnamed: 45_level_1': '', 'Unnamed: 46_level_1': '', 'Unnamed: 47_level_1': ''}, level=1)
df['meds'] = df['med'].apply(lambda r: tuple(
    c for c in df['med'].columns if r[c] == 1), axis=1)  # tuples of mediators
# tuples of ws pairs
df['ws'] = list(zip(df[('params', 'W1')], df[('params', 'W2')]))
df['ts'] = list(zip(df[('game', 't')], df[('game', 's')]))
df['coop_conv'] = (df[('agents', 'coop_freq')] == 1).astype(int)
df_mean = df.groupby(["meds", "ws"]).mean()
df_mean_ts = df.groupby(["meds", ("params", "W1"), "ts"]).mean()
# %%
w1 = w1s[0]
df_pivoted = df_mean_ts.loc[("NO_MED",)].loc[w1].pivot(
    index=[("game", "s")], columns=[("game", "t")]).sort_index(ascending=False)
ss, ts = ['%.2f' % elem for elem in df_pivoted[("agents", "coop_freq")].index], [
    '%.2f' % elem for elem in df_pivoted[("agents", "coop_freq")].columns]
plot_heatmap(df_pivoted[("agents", "coop_freq")], vmin=0, vmax=1, yticklabels=ss,
             xticklabels=ts, title=f"Avg. final #cooperators", interpolate=False)

# %%
plot_heatmap(df_pivoted["coop_conv"], vmin=0, vmax=1, yticklabels=ss, xticklabels=ts,
             title=f"Avg. #cooperators convergence", interpolate=False)
# %%
# %%
df[['net', 'agents', 'params', 'coop_conv', 'ts']][df['ts'] == (2, -1)]
[r for r in ts_res[0] if r['params']['t'] == 2 and r['params']['s'] == -1]
