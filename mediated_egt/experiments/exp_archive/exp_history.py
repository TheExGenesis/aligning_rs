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
from optimized.cyFns import *

episode_n = 1000
medSet = [0, 1]
saveHistory = True
save = False
n_trials = 1
N = 500
W1 = 1
W2 = 0
ts = (2, -1)
beta = 0.005
k = 30
M = 5
dilemma = cy_makeTSDilemma(2, -1)
medStrats = cy_initMedStrats(N, medSet)
strats = cy_initStrats(N)
history = []
x = crandint(0, N-1)
graph = initUniformRandomGraph(
    N=N, k=(k if k else _k))
# graph1, history1 = cy_runEvolutionCompetitionEp(N, beta, W1, W2, dilemma, graph, medStrats, strats, history, x, saveHistory=saveHistory)


res = cy_genericRunEvolution(N, episode_n, W1, W2, dilemma, medStrats,
                             strats, beta, graph, k, history, saveHistory=saveHistory)
