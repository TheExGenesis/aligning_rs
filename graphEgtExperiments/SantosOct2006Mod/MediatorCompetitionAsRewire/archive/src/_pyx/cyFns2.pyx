from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log, exp

cdef int crandint(int lower, int upper) except -1:
    return (rand() % (upper - lower + 1)) + lower
    
cdef inline float float_max(float a, float b): return a if a >= b else b
cdef inline float float_min(float a, float b): return a if a <= b else b


cdef inline float clamp(float val, float minval, float maxval): return float_max(minval, float_min(val, maxval))
cdef inline float clip(float val, float minval): return float_max(minval, val)


cdef float fermi2(float beta, float fitness_diff):
    cdef float clipDiff = clip(fitness_diff, 0)
    cdef float exponent = -1 * beta * clipDiff
    cdef float exponential = exp(exponent)
    # cdef float exponential = np.exp(exponent, dtype=np.float64)
    cdef float p = 1. / (1. + exponential)
    return p
