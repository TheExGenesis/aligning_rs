# Evaluate cython implementation, numba implementation, and numba + networkx implementation
# %%
# evaluate performance of cython implementation
%load_ext cython
%load_ext line_profiler

#%%
from evolution import *
import matplotlib.pyplot as plt
from itertools import chain
from utils import transposeList
from plots import *
from egt_io import *
from defaultParams import _episode_n, _N, _W, _W2, _k, _T, _S
from defaultParams import *
from mediators import *
from mediators import _medSet
from itertools import product, combinations
from functools import reduce
from dataframe import makeEntry2, makeColumns
import seaborn as sns
from math import inf
from optimized.cyFns import *

# %%
# %%cython
# setup cy ep
N = 500
beta = 0.005
k = 30
medSet = [5, 6, 7, 8]
# medStrats = initMedStrats(N, medSet)
ts = (2, -1)
x = crandint(0, N-1)
W1 = 1
W2 = 0.1
graph = initUniformRandomGraph(N, k=k)
graph.set_fast_edge_removal(True)
history = []
dilemma = cy_makeTSDilemma(2, -1)
medStrats = cy_initMedStratsSmall(N, medSet, 0)
strats = cy_initStrats(N)

cy_runEvolutionCompetitionEp(N, beta, W1, W2, dilemma, graph,
                          medStrats, strats, history, crandint(0, N-1), saveHistory=False)

# %%
# time cython implementation
%timeit cy_runEvolutionCompetitionEp(N, beta, W1, W2, dilemma, graph, medStrats, strats, history, crandint(0, N-1), saveHistory=False)
# around 28.3 µs ± 3.21 µs 



# profiling cython implementation
# %%
# can't profile cython function
%lprun -f  cy_runEvolutionCompetitionEp cy_runEvolutionCompetitionEp(N, beta, inf, 0, dilemma, graph, medStrats, strats, history, x, saveHistory=False)


# %%
%%time 
cy_runEvolutionCompetitionEp(N, beta, W1, W2, dilemma, graph, medStrats, strats, history, crandint(0, N-1), saveHistory=False)



# %%
# evaluate performance of numba implementation
# %%
from evolution2 import *
from mediators2 import *
from sampling2 import *
# %%
N = 500
beta = 0.005
k = 30
medSet = [5, 6, 7, 8]
strats = initStrats(N)
medStrats = initMedStratsSmall(N, medSet, 0)
# medStrats = initMedStrats(N, medSet)
ts = (2, -1)
dilemma = makeTSDilemma(2, -1)
x = np.random.randint(0, N-1)
W1 = 1
W2 = 0.1
graph = initUniformRandomGraph(N, k=k)
graph.set_fast_edge_removal(True)
history = []
runEvolutionCompetitionEp(N, beta, W1, W2, dilemma, graph,
                          medStrats, strats, history, x, saveHistory=False)
#%%
%timeit runEvolutionCompetitionEp(N, beta, W1, W2, dilemma, graph, medStrats, strats, history, crandint(0, N-1), saveHistory=False)
# around 21.2 µs ± 3.74 µs  

# %%
# evaluate performance of numba + networkx implementation
# %%
from evolution_nx import *
from mediators_nx import *
from sampling_nx import *
# %%
N = 500
beta = 0.005
k = 30
medSet = [5, 6, 7, 8]
strats = initStrats(N)
medStrats = initMedStratsSmall(N, medSet, 0)
# medStrats = initMedStrats(N, medSet)
ts = (2, -1)
dilemma = makeTSDilemma(2, -1)
x = np.random.randint(0, N-1)
W1 = 1
W2 = 0.1
graph = initUniformRandomGraph(N, k=k)
graph_nx = nx.Graph(spectral.adjacency(graph))
graph = nx.relabel.convert_node_labels_to_integers(graph_nx)
history = []
runEvolutionCompetitionEp(N, beta, W1, W2, dilemma, graph,
                          medStrats, strats, history, x, saveHistory=False)
#%%
%timeit runEvolutionCompetitionEp(N, beta, W1, W2, dilemma, graph, medStrats, strats, history, crandint(0, N-1), saveHistory=False)
# around 22.6 µs ± 2.91 µs
# %%
