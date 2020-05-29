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

@njit
def generate_alphas(velocity_bins, theta_bin_n, phi_bin_n, radius):
    '''generate evenly spaced alpha0 and alpha1 vectors.
    phi_bin_n refers to the maximum number of phi_bins; to keep the bins evenly distributed on the
    sphere phi bin count would decrease moving out from the equator. Velocity bins are assumed to
    be bin EDGES, in SI units.

    Vectors are generated as a list, assuming entry time is zero.
    '''
    points_on_sphere = []
    velocity_bin_centres = velocity_bins[:-1] + np.diff(velocity_bins)/2
    theta_bins = np.linspace(0, np.pi, theta_bin_n+1)
    theta_bin_centres = theta_bins[:-1] + np.diff(theta_bins)/2
    for t in theta_bin_centres:
        phi_bin_n_cur = int(np.round(phi_bin_n/np.sin(t)))
        phi_bins = np.linspace(0, 2*np.pi, phi_bin_n_cur+1)
        phi_bins_centres = phi_bins[:-1] + np.diff(phi_bins)/2
        for p in phi_bins_centres:
            points_on_sphere.append([radius*np.sin(t)*np.cos(p),
                                     radius*np.sin(t)*np.sin(p),
                                     radius*np.cos(t)
                                    ])



def transform(data,):
    '''Takes time series data as an input and generates a signal value based on
    entry and exit 4-vectors on a sphere extended in time. Returns the signal
    value and 4-vectors. Refer to Qin's note for a much more detailed
    explanation.'''
