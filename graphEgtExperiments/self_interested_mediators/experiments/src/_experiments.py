# %%
import line_profiler
from Cython.Compiler.Options import get_directive_defaults
directive_defaults = get_directive_defaults()

directive_defaults['linetrace'] = True
directive_defaults['binding'] = True
%load_ext line_profiler
%load_ext Cython
# %%
%%cython -a
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
cimport numpy as cnp
import sys
from pathlib import Path
from updates import calcStrategyUpdate, updateStrat, calcStructuralUpdate, updateTies, calcMedUpdate, updateMed
from sampling import sampleNeighbor
from games import pairCumPayoffs, makeTSDilemma, nodeCumPayoffs
from mediators import *
from mediators import _medSet
from init import initUniformRandomGraph, initPayoffs, initStrats
from defaultParams import _episode_n, _N, _W, _W2, _k, _T, _S
from defaultParams import *
from functools import partial
from math import floor, inf
from collections import Counter
from copy import deepcopy
import pandas as pd
from updates import updateTies, calcStructuralUpdate, TieUpdate
from sampling import sampleNeighbor, isLonely
from egt_io import saveRes, makeCompetitionName
from evolution import *
import numpy as np
from math import inf
from time import time
import array
from optimized.cyFns import *
from cpython cimport array
from libc.math cimport exp
from libc.math cimport log, exp
from libc.stdlib cimport rand, RAND_MAX
import cython

#%%
N=500
graph = initUniformRandomGraph(N)
from graph_tool.spectral import adjacency
x=crandint(0,N-1)
y=crandint(0,N-1)
mat = adjacency(graph).tolil()
rows = mat.rows.copy() 
#%%
%timeit np.random.choice(graph.get_all_neighbors(x))
%timeit graph.get_all_neighbors(x)[crandint(0, graph.get_out_degrees([x])[0]-1)]
%timeit np.random.choice(graph.get_out_neighbors(x))
%timeit graph.get_out_neighbors(x)[crandint(0, graph.get_out_degrees([x])[0]-1)]
%timeit np.random.choice(rows[x])
%timeit rows[x][crandint(0, graph.get_out_degrees([x])[0]-1)]
# %%
%timeit graph.get_all_neighbors(x)
%timeit adjacency(graph).tolil().rows[x]
