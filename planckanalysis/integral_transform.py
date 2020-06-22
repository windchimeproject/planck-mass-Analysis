'''integral transform module.'''
import json
from numba import njit
import numpy as np
from tqdm import tqdm, trange

@njit
def delta_response(delta_t):
    '''the response function is defined in terms of the number of samples between
    the source time and the observer time (ie. time of current sample).
    '''
    if delta_t == 0:
        return 1
    return 0

@njit
def signal_function(vector_delta, response_func=delta_response, sensor_radius=1e-3):
    '''signal template function. Should take into account response function!
    vector_delta.shape == (n, 3), for n inputs.
    '''
    out = np.zeros((vector_delta.shape[0], 3))
    for i in range(vector_delta.shape[0]):
        denom = max([(vector_delta[i, 0]**2 + vector_delta[i, 1]**2 + vector_delta[i, 2]**2)**(3/2),
                     sensor_radius**3])
        out[i, 0] = (response_func(vector_delta[i, 3]) * vector_delta[0, i]/denom)
        out[i, 1] = (response_func(vector_delta[i, 3]) * vector_delta[1, i]/denom)
        out[i, 2] = (response_func(vector_delta[i, 3]) * vector_delta[2, i]/denom)
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
    for theta in theta_bin_centres:
        phi_bin_n_cur = int(np.round(phi_bin_n*np.sin(theta)))
        phi_bins = np.linspace(0, 2*np.pi, phi_bin_n_cur+1)
        phi_bins_centres = phi_bins[:-1] + np.diff(phi_bins)/2
        for phi in phi_bins_centres:
            points_on_sphere.append([radius*np.sin(theta)*np.cos(phi),
                                     radius*np.sin(theta)*np.sin(phi),
                                     radius*np.cos(theta)
                                    ])
    out = []
    points_on_sphere2 = points_on_sphere.copy()
    for point in points_on_sphere:
        for point2 in points_on_sphere2:
            for vel in velocity_bin_centres:
                x_0 = point[0]
                y_0 = point[1]
                z_0 = point[2]
                x_1 = point2[0]
                y_1 = point2[1]
                z_1 = point2[2]
                length = np.sqrt(
                    (x_1-x_0)**2 +
                    (y_1-y_0)**2 +
                    (z_1-z_0)**2
                )
                if length > 0:
                    out.append([
                        x_0,
                        y_0,
                        z_0,
                        0,
                        x_1,
                        y_1,
                        z_1,
                        length/vel,
                    ])
    return np.array(out)

def generate_adc_lookup_table(acceleration_bin_edges):
    '''Takes on an array of acceleration bin edges in order to create a dictionary with 
    keys composed of ADC numbers and values of average accleration 
    '''
    i = 1
    lookup_dict = {}
    lookup_dict[0] = float("-inf")
    for s in range (0,len(acceleration_bin_edges)-1):
        lookup_dict[i] = (acceleration_bin_edges[s]+acceleration_bin_edges[s+1])/2
        i += 1
    lookup_dict[65535] = float("inf")
    return lookup_dict


def adc_readout_to_accel(data, lookup_dict):
    '''converts adc values to accelerations'''
    out = np.zeros(data.shape)
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            out[i, j] = lookup_dict[value]
    return out


def transform(times, accels, timesteps, timestep_indices, alphas, sensors_pos, response_length=2):
    '''Takes time series data as an input and generates a signal value based on
    entry and exit 4-vectors on a sphere extended in time. Returns the signal
    value and 4-vectors. Refer to Qin's note for a much more detailed
    explanation.

    accels is a list of accelerations.
    
    sensor_dict is a list of sensor positions, in the same order as accels.
    '''
    S = []
    S_norm = []
    alpha0_x = []
    alpha0_y = []
    alpha0_z = []
    alpha0_t = []
    alpha1_x = []
    alpha1_y = []
    alpha1_z = []
    alpha1_t = []
    adc_timestep_size = times[1] - times[0]
    for i,start_time in enumerate(timesteps[:-1]):
        for alpha_index in trange(alphas.shape[0]):
            alpha_pair = alphas[alpha_index,:]
            start_index = timestep_indices[i]
            dir_vector = np.array([
                alpha_pair[4] - alpha_pair[0],
                alpha_pair[5] - alpha_pair[1],
                alpha_pair[6] - alpha_pair[2],
            ])
            initial_pos = np.array([alpha_pair[0], alpha_pair[1], alpha_pair[2]])
            dir_vector_step = dir_vector/(alpha_pair[7] - alpha_pair[3]) * adc_timestep_size
            n_steps = int(np.ceil((alpha_pair[7] - alpha_pair[3])/adc_timestep_size))
            particle_pos_arr = np.array(
                [initial_pos + j*dir_vector_step for j in range(n_steps+response_length-1)]
            )
            track_times = np.array(
                [start_time + j*adc_timestep_size for j in range(n_steps+response_length-1)]
            )
            # track_indices = np.array(
            #     [start_index + j for j in range(n_steps+response_length-1)]
            # )
            S_this_track = 0
            for j in range(n_steps):
                for sens_num, sensor_pos in enumerate(sensors_pos):
                    vector_delta = np.zeros((response_length, 4))
                    for k in range(response_length):
                        vector_delta[k, 0] = (particle_pos_arr[j - k + response_length - 1][0] -
                                              sensor_pos[0])
                        vector_delta[k, 1] = (particle_pos_arr[j - k + response_length - 1][1] -
                                              sensor_pos[1])
                        vector_delta[k, 2] = (particle_pos_arr[j - k + response_length - 1][2] -
                                              sensor_pos[2])
                        vector_delta[k, 3] = (track_times[j - k + response_length - 1] -
                                              track_times[j + response_length - 1])
                    expected_signal_from_sensor = signal_function(vector_delta)
                    signal_from_sensor = accels[sens_num][j:j+response_length]
                    S_this_track += np.einsum(
                        'ij,ij->',expected_signal_from_sensor,signal_from_sensor
                    )
            S.append(S_this_track)
            if n_steps > 0:
                S_norm.append(S_this_track/n_steps)
            else:
                S_norm.append(0)
            alpha0_x.append(alpha_pair[0])
            alpha0_y.append(alpha_pair[1])
            alpha0_z.append(alpha_pair[2])
            alpha0_t.append(alpha_pair[3])
            alpha1_x.append(alpha_pair[4])
            alpha1_y.append(alpha_pair[5])
            alpha1_z.append(alpha_pair[6])
            alpha1_t.append(alpha_pair[7])
    structured_array = np.zeros(len(S), dtype=[
        ('S', 'f8'),
        ('S_norm', 'f8'),
        ('alpha0_x', 'f8'),
        ('alpha0_y', 'f8'),
        ('alpha0_z', 'f8'),
        ('alpha0_t', 'f8'),
        ('alpha1_x', 'f8'),
        ('alpha1_y', 'f8'),
        ('alpha1_z', 'f8'),
        ('alpha1_t', 'f8'),
    ])
    structured_array['S'] = S
    structured_array['S_norm'] = S_norm
    structured_array['alpha0_x'] = alpha0_x
    structured_array['alpha0_y'] = alpha0_y
    structured_array['alpha0_z'] = alpha0_z
    structured_array['alpha0_t'] = alpha0_t
    structured_array['alpha1_x'] = alpha1_x
    structured_array['alpha1_y'] = alpha1_y
    structured_array['alpha1_z'] = alpha1_z
    structured_array['alpha1_t'] = alpha1_t
    return structured_array