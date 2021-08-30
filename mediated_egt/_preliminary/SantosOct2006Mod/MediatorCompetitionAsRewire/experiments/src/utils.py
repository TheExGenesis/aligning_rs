# %%
import functools
from collections.abc import Iterable


def flatten(x):
    if isinstance(x, Iterable):
        return sum([flatten(i) for i in x], [])
    return [x]


'''Utils'''


def transposeList(l):
    return list(map(list, zip(*l)))


def orderTsMatrixPlot(ts, M):
    return [y for x in [ts[(M-1-i)::M] for i in range(M)] for y in x]


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def pipe(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), reversed(functions), lambda x: x)
