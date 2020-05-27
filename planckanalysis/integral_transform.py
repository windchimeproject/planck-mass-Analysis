'''integral transform module.'''
from numba import njit
import numpy as np

@njit
def delta_response(delta_t):
    '''the response function is defined in terms of the number of samples between
    the source time and the observer time (ie. time of current sample).
    '''
    if delta_t == 0:
        return 1
    return 0

@njit
def signal_function(vector_delta, response_func=delta_response):
    '''signal template function. Should take into account response function!
    vector_delta.shape == (4,n), for n inputs.
    '''
    out = np.zeros((3, vector_delta.shape[1]))
    for i in range(vector_delta.shape[1]):
        denom = (vector_delta[1, i]**2 + vector_delta[2, i]**2 + vector_delta[3, i]**2)**(3/2)
        out[0, i] = (response_func(vector_delta[0, i]) *
                     vector_delta[1, i]/denom
                    )
    return out


def generate_alphas():
    '''generate evenly spaced alpha0 and alpha1 vectors. (Even on 4D cylinder)'''


def transform(data,):
    '''Takes time series data as an input and generates a signal value based on
    entry and exit 4-vectors on a sphere extended in time. Returns the signal
    value and 4-vectors. Refer to Qin's note for a much more detailed
    explanation.'''
