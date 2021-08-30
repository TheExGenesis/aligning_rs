# cythonized functions for structural update which don't really seem to speed up things

def isLonely(graph, x):
    return graph.vertex(x).out_degree() <= 1





def cy_useMed(int medStrat, graph, int[:] strats, int y, int x):
    return int2Med[medStrat](graph, strats, y, x)


def cy_rewireEdge(graph, int x, int y, int z):
    graph.remove_edge(graph.edge(x, y))
    graph.add_edge(x, z)
    return graph

def cy_updateTies(graph, Update tieUpdate):
    return cy_rewireEdge(graph, tieUpdate.x, tieUpdate.old, tieUpdate.new)

def cy_decideRewire(graph, int [:] strats, int x, int y, int [:] medStrats):
    cdef Update tieUpdate
    if isLonely(graph, y):
        tieUpdate = [MEDIATOR, x, y, y]
        return tieUpdate  # enforcing graph connectedness
    z = cy_useMed(medStrats[x], graph, strats, y, x)
    if not z:
        tieUpdate = [MEDIATOR, x, y, y]
        return tieUpdate  # enforcing graph connectedness
    tieUpdate = [MEDIATOR, x, y, z]
    return tieUpdate
# # Decides whether a rewire should happen based on x and y's strats. If x is satisfied, nothing happens


def cy_calcStructuralUpdate(graph, int[:] strats, int x, int y, float p, int[:] medStrats):
    cdef Update tieUpdate
    cdef bint doRewire
    if (strats[x] == C and strats[y] == D):
        doRewire = rand()/RAND_MAX < p
        if doRewire:
            return cy_decideRewire(graph, strats, x, y, medStrats)
        else:
            tieUpdate = [MEDIATOR, x, y, y]
            return tieUpdate
    elif (strats[x] == D and strats[y] == D):
        keepX = rand()/RAND_MAX < p
        if keepX:
            return cy_decideRewire(graph, strats, x, y, medStrats)
        else:
            return cy_decideRewire(graph, strats, y, x, medStrats)
    tieUpdate = [MEDIATOR, x, y, y]
    return tieUpdate
# # Applies a tie update to the graph

def full_cy_runEvolutionCompetitionEp(int N, float beta, float W, float W2, float[:, :, :] dilemma, graph, int[:] medStrats, int[:] strats, history, int _x, bint saveHistory=False):
    cdef int x = _x
    cdef long[:] neighs_x = graph.get_all_neighbors(x)
    cdef int _y = neighs_x[crandint(0, len(neighs_x)-1)]  # sample neighbor
    # cdef int x = int(_x)
    # cdef int y = int(_y)
    cdef int y = _y
    cdef long[:] neighs_y = graph.get_all_neighbors(y)
    cdef float px = cy_nodeCumPayoffs(dilemma, neighs_x, strats, _x)
    cdef float py = cy_nodeCumPayoffs(dilemma, neighs_y, strats, _y)
    cdef float p = cy_fermi(beta, py - px)
    cdef float r = rand() / RAND_MAX
    cdef bint doMedUpdate = r * (1+W2) > 1
    if doMedUpdate:
        medUpdate = cy_calcMedUpdate(medStrats, x, y, p)
        if saveHistory:
            history.append(medUpdate)
        medStrats = cy_updateMed(medStrats, medUpdate)
    else:
        doStratUpdate = (rand() / RAND_MAX) * (1+W) <= 1
        if doStratUpdate:
            stratUpdate = cy_calcStrategyUpdate(strats, x, y, p)
            if saveHistory:
                history.append(stratUpdate)
            strats = cy_updateStrat(strats, stratUpdate)
        else:
            graphUpdate = cy_calcStructuralUpdate(graph, strats, _x, _y, p, medStrats)
            if saveHistory:
                history.append(graphUpdate)
            graph = cy_updateTies(graph, graphUpdate)
    return strats, graph, history

