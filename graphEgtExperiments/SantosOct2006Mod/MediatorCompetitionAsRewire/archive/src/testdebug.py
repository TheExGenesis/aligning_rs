medSet = [0, 1]
n_trials = 1
N = 500
episode_n = 100000
W1 = 1
W2 = 1
ts = (2, -1)
beta = 0.005
k = 30

_strats = cy_initStrats(N)[:]
_medStrats = cy_initMedStrats(N, medSet)[:]
dilemma = cy_makeTSDilemma(*ts)
_graph = initUniformRandomGraph(N=N, k=(k if k else _k))


history = []
initialStrats = _strats[:]
strats = _strats[:]
initialMedStrats = _medStrats[:]
medStrats = _medStrats[:]
totalPayoffs = initPayoffs(N)
graph = _graph
i = 0
dilemma = cy_makeTSDilemma(*ts)

x = crandint(0, N-1)
strats, graph, history = cy_runEvolutionCompetitionEp(
    N, beta, W1, W2, dilemma, graph, medStrats, strats, history, x, False)


res = cy_genericRunEvolution(N, 50000, W1, W2, dilemma, _medStrats,
                             _strats, beta, deepcopy(_graph), k, history, saveHistory=False)
