
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log, exp
cimport numpy as cnp
import numpy as np


# Utils
cpdef int crandint(int lower, int upper) except -1:
    return (rand() % (upper - lower + 1)) + lower

cpdef float randFloat() except -1:
    cdef float r = (rand() % RAND_MAX) / float(RAND_MAX)
    return r


cdef inline float float_max(float a, float b): return a if a >= b else b
cdef inline float float_min(float a, float b): return a if a <= b else b


cdef inline float clamp(float val, float minval, float maxval): return float_max(minval, float_min(val, maxval))
cdef inline float clip(float val, float minval): return float_max(minval, val)


cpdef float cy_fermi(float beta, float fitness_diff):
    cdef float clipDiff = clip(fitness_diff, 0)
    cdef float exponent = -1 * beta * clipDiff
    cdef float exponential = exp(exponent)
    cdef float p = 1. / (1. + exponential)
    return p


# init

# init strats

intToStrat = {1:"C", 0:"D"}
cpdef enum Strat: D, C

cpdef int[:] cy_initStrats(int N):
    cdef int[:] strats
    strats = np.array([crandint(0,1) for _ in range(N)], dtype=np.int32)
    return strats


# init meds
'''Mediator strats'''
# GOOD_MED = 'GOOD_MED'
# BAD_MED = 'BAD_MED'
# NO_MED = 'NO_MED'
# RANDOM_MED = 'RANDOM_MED'
# FAIR_MED = 'FAIR_MED'

# _medSet = [NO_MED, GOOD_MED, FAIR_MED, RANDOM_MED, BAD_MED]

# cpdef enum MedStrat: NO_MED, GOOD_MED, FAIR_MED, RANDOM_MED, BAD_MED
# int2Med = {0:NO_MED, 1:GOOD_MED, 2:FAIR_MED, 3:RANDOM_MED, 4:BAD_MED}


cpdef int[:] cy_initMedStrats(int N, medSet):
    cdef int[:] medStrats
    medStrats = np.array([medSet[crandint(0, len(medSet)-1)] for i in range(N)], dtype=np.intc)
    return medStrats



# medDict = {NO_MED: useNoMed, GOOD_MED: useGoodMed,
#            BAD_MED: useBadMed, RANDOM_MED: useRandomMed, FAIR_MED: useFairMed}

cpdef enum UpdateType: MEDIATOR, STRAT, REWIRE
ctypedef struct Update:
    int updateType
    int x
    int old
    int new


cpdef Update cy_calcMedUpdate(int[:] medStrats, int x, int y, float p):
    cdef bint doChangeStrat = rand()/RAND_MAX < p
    cdef int old = medStrats[x]
    cdef int new
    if doChangeStrat:
        new = medStrats[y]
    else:
        new = old
    cdef Update medUpdate = [MEDIATOR, x, old, new]
    return medUpdate


cpdef int[:] cy_updateMed(int[:] medStrats, Update update):
    medStrats[update.x] = update.new
    return medStrats

# strat updates
cpdef Update cy_calcStrategyUpdate(int[:] strats, int x, int y, float p):
    cdef bint doChangeStrat = rand()/RAND_MAX < p
    cdef int old = strats[x]
    cdef int new
    if doChangeStrat:
        new = strats[y]
    else:
        new = old
    cdef Update stratUpdate = [STRAT, x, old, new]
    return stratUpdate
# Applies a strategy update to the graph


cpdef int[:] cy_updateStrat(int[:] strats, Update update):
    strats[update.x] = update.new
    return strats

# structural updates



# games

# game params
_R = 1.0
_P = 0.0
_T = 2.0  # T€[0,2]
_S = -1.0  # S€[-1,1]

cpdef float[:,:,:] cy_makeDilemma(float R=_R, float P=_P, float T=_T, float S=_S):
    cdef float[:,:,:] dilemma_view = np.array([[[P,P], [T,S]], [[S,T], [R,R]]], dtype=cnp.dtype("f"))
    return dilemma_view
cpdef float[:,:,:] cy_makeTSDilemma(float t, float s):
    return cy_makeDilemma(_R,_P,t,s)



cpdef float cy_playDilemmaP1(float[:,:,:] dilemma, int[:] strats, int id1, int id2):
    cdef int s1 = strats[id1]
    cdef int s2 = strats[id2]
    return dilemma[s1][s2][0]


cpdef float cy_nodeCumPayoffs(float[:,:,:] dilemma, long[:] neighbors, int[:] strats, int x):
    cdef float totalPayoff=0
    #cdef long[:] neighbors = graph.get_all_neighbors(x) 
    cdef int i = 0
    for i in range(neighbors.shape[0]):
        totalPayoff += cy_playDilemmaP1(dilemma, strats, x, neighbors[i])
    return totalPayoff


