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


# %%
med = 1
episode_n = 400000
W1 = 2
saveHistory = True
M = 5
gameParams = genTSParams(M)
medSet = [med]
N = 500
W2 = 0
beta = 0.005
k = 30
runs = {ts: cy_runCompetitionExperiment(N=N, episode_n=episode_n, W1=W1, W2=W2, ts=ts,
                                        beta=beta, k=k, saveHistory=saveHistory, history=[], medSet=medSet) for ts in gameParams}
c = coop([runs])
df = pd.DataFrame(zip(list(c.values()), *transposeList(list(c.keys()))),
                  columns=["coops", 't', 's']).pivot('s', 't', "coops").iloc[::-1]
sns.heatmap(df, annot=True, cbar=True, xticklabels=2,
            yticklabels=2, vmin=0, vmax=1, ax=plt.gca()).set(title=f"{int2MedName[med]}, W1={k}")
