'''integral transform module.'''
import json
from numba import njit
import numpy as np
from scipy import signal
from tqdm import tqdm, trange


def signal_function(vector_delta, lin_resp, sensor_radius=1e-4):
    '''signal template function. Should take into account response function!
    vector_delta.shape == (n, 3), for n inputs.
    '''
    out = np.zeros((vector_delta.shape[0], 3))
    convolved_list = []

    denoms = np.maximum(np.linalg.norm(vector_delta, axis=1) ** 3, np.repeat(sensor_radius ** 3, vector_delta.shape[0]))
    out[:, 0] = (vector_delta[:, 0] / denoms)
    out[:, 1] = (vector_delta[:, 1] / denoms)
    out[:, 2] = (vector_delta[:, 2] / denoms)

    convolved_list.append(signal.convolve(out[:, 0], lin_resp, mode='full'))
    convolved_list.append(signal.convolve(out[:, 1], lin_resp, mode='full'))
    convolved_list.append(signal.convolve(out[:, 2], lin_resp, mode='full'))
    convolved_signal = np.array(convolved_list).T

    return convolved_signal

def Time_Analysis_alphas(vel, entry_vecs, exit_vecs):
    velocity = vel
    alphas = []

    for vel_p in velocity:
        x_0 = entry_vecs[0][0]
        y_0 = entry_vecs[1][0]
        z_0 = entry_vecs[2][0]
        x_1 = exit_vecs[0][0]
        y_1 = exit_vecs[1][0]
        z_1 = exit_vecs[2][0]
        length = np.sqrt(
            (x_1 - x_0) ** 2 +
            (y_1 - y_0) ** 2 +
            (z_1 - z_0) ** 2
        )
        alphas.append([
            x_0,
            y_0,
            z_0,
            0,
            x_1,
            y_1,
            z_1,
            length / vel_p,
        ])

    return alphas


def Velocity_Analysis_alphas(entry_vecs, exit_vecs, num_bin = 250):
    velocity_bins = np.linspace(1e5, 7e5, num_bin)
    velocity_bin_centres = velocity_bins[:-1] + np.diff(velocity_bins) / 2
    alphas = np.zeros((len(velocity_bin_centres), 8))
    for i, vel_p in enumerate(velocity_bin_centres):
        x_0 = entry_vecs[0]
        y_0 = entry_vecs[1]
        z_0 = entry_vecs[2]
        x_1 = exit_vecs[0]
        y_1 = exit_vecs[1]
        z_1 = exit_vecs[2]
        length = np.sqrt(
            (x_1 - x_0) ** 2 +
            (y_1 - y_0) ** 2 +
            (z_1 - z_0) ** 2
        )
        alphas[i] = np.array([
            x_0,
            y_0,
            z_0,
            0,
            x_1,
            y_1,
            z_1,
            length / vel_p,
        ])

    return alphas


def py_ang(v1, v2):
    # Returns the angle in radians between vectors 'v1' and 'v2'
    cosdata = np.dot(v1, v2)
    denominator = np.linalg.norm(v1) * np.linalg.norm(v2)
    cosang = cosdata / denominator

    return cosang


def Theta_Analysis_alphas(vel, entry_vecs, exit_vecs, radius=5.2):
    thetas = np.linspace(0, 90, 90)
    theta_cos_val = []
    vel_p = vel
    alpharange = []
    alpha1 = exit_vecs - entry_vecs
    alpha = np.array([alpha1[0][0], alpha1[1][0], alpha1[2][0]])  # redefine/restructure alpha
    alphas = []

    for i in thetas:
        x_0 = entry_vecs[0]
        y_0 = entry_vecs[1]
        z_0 = entry_vecs[2]

        vec = np.random.randn(3, 1)
        coordinates = vec / np.linalg.norm(vec, axis=0)

        x_1 = coordinates[0][0] * radius
        y_1 = coordinates[1][0] * radius
        z_1 = coordinates[2][0] * radius
        length = np.sqrt(
            (x_1 - x_0) ** 2 +
            (y_1 - y_0) ** 2 +
            (z_1 - z_0) ** 2
        )
        ydiff = y_1 - y_0
        xdiff = x_1 - x_0
        zdiff = z_1 - z_0

        alphas.append([
            x_0,
            y_0,
            z_0,
            0,
            x_1,
            y_1,
            z_1,
            length / vel_p,
        ])

        alphaprime = np.array([xdiff, ydiff, zdiff])
        alpharange.append(alphaprime)

        # costheta = np.dot(alpha, alphaprime)/(np.linalg.norm(alpha))*(np.linalg.norm(alphaprime))
        # theta = np.arccos(costheta)
        costheta = py_ang(alphaprime.T, alpha.T)
        theta_cos_val.append(costheta)

    return alphas, theta_cos_val


def generate_adc_lookup_table(acceleration_bin_edges):
    '''Takes on an array of acceleration bin edges in order to create a dictionary with
    keys composed of ADC numbers and values of average acceleration
    '''
    i = 1
    lookup_dict = {}
    for s in range(0, len(acceleration_bin_edges) - 1):
        lookup_dict[i] = (acceleration_bin_edges[s] + acceleration_bin_edges[s + 1]) / 2
        i += 1
    lookup_dict[0] = acceleration_bin_edges[0]
    lookup_dict[65535] = acceleration_bin_edges[-1]
    return lookup_dict


def adc_readout_to_accel(data, lookup_dict, sensitivity=1):
    '''converts adc values to accelerations'''
    out = np.zeros(data.shape)
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            out[i, j] = lookup_dict[value]
    return out


def transform_temp(times, timesteps, timestep_indices, alphas, sensors_pos, lin_resp):
    '''Takes time series data as an input and generates a signal value based on
    entry and exit 4-vectors on a sphere extended in time. Returns the signal
    value and 4-vectors. Refer to Qin's note for a much more detailed
    explanation.

    accels is a list of accelerations.

    sensor_dict is a list of sensor positions, in the same order as accels.
    '''

    expected_signal_from_sensor = []
    sens_id = []
    start_ind_id = []
    rng_sens_num_id = []
    start_tm_id = []
    n_stp_id = []

    alpha0_x = []
    alpha0_y = []
    alpha0_z = []
    alpha0_t = []
    alpha1_x = []
    alpha1_y = []
    alpha1_z = []
    alpha1_t = []
    steps = []

    adc_timestep_size = times[1] - times[0]
    response_length = len(lin_resp)
    for i, start_time in enumerate(tqdm(timesteps)):
        start_tm_id.append(start_time)
        for alpha_index in range(alphas.shape[0]):
            alpha_pair = alphas[alpha_index, :]
            start_index = timestep_indices[i]
            dir_vector = np.array([
                alpha_pair[4] - alpha_pair[0],
                alpha_pair[5] - alpha_pair[1],
                alpha_pair[6] - alpha_pair[2],
            ])
            initial_pos = np.array([alpha_pair[0], alpha_pair[1], alpha_pair[2]])

            dir_vector_step = dir_vector / (alpha_pair[7] - alpha_pair[3]) * adc_timestep_size
            n_steps = min(int(np.ceil((alpha_pair[7] - alpha_pair[3]) / adc_timestep_size)),
                          len(times[times > start_time]))

            particle_pos_arr = np.array([initial_pos + j * dir_vector_step for j in range(n_steps)])
            track_times = np.array([start_time + j * adc_timestep_size for j in range(n_steps)])

            start_ind_id.append(start_index)
            n_stp_id.append(n_steps)

            if n_steps > 0:
                rng_sens_num = 0
                for sens_num, sensor_pos in enumerate(sensors_pos):
                    vector_delta = np.zeros((n_steps, 4))
                    rng_sens_num += 1
                    sens_id.append(sens_num)
                    for j in range(n_steps):
                        vector_delta[j, 0] = (particle_pos_arr[j][0] - sensor_pos[0])
                        vector_delta[j, 1] = (particle_pos_arr[j][1] - sensor_pos[1])
                        vector_delta[j, 2] = (particle_pos_arr[j][2] - sensor_pos[2])
                        ##vector_delta[j, 3] = (track_times[j] - #step_value) # Something like this can be used for analysis of response time
                    expected_signal_from_sensor.append(np.array(signal_function(vector_delta, lin_resp, adc_timestep_size)))
                rng_sens_num_id.append(rng_sens_num)
            else:
                rng_sens_num_id.append(0)
            alpha0_x.append(alpha_pair[0])
            alpha0_y.append(alpha_pair[1])
            alpha0_z.append(alpha_pair[2])
            alpha0_t.append(alpha_pair[3] + start_time)
            alpha1_x.append(alpha_pair[4])
            alpha1_y.append(alpha_pair[5])
            alpha1_z.append(alpha_pair[6])
            alpha1_t.append(alpha_pair[7] + start_time)
            steps.append(n_steps)
    structured_array = np.zeros(len(steps), dtype=[
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
        ('steps', 'i4'),
    ])
    structured_array['alpha0_x'] = alpha0_x
    structured_array['alpha0_y'] = alpha0_y
    structured_array['alpha0_z'] = alpha0_z
    structured_array['alpha0_t'] = alpha0_t
    structured_array['alpha1_x'] = alpha1_x
    structured_array['alpha1_y'] = alpha1_y
    structured_array['alpha1_z'] = alpha1_z
    structured_array['alpha1_t'] = alpha1_t
    structured_array['steps'] = steps

    return expected_signal_from_sensor, sens_id, start_ind_id, rng_sens_num_id, start_tm_id, n_stp_id, structured_array

def transform_calc(expected_signal_from_sensor, sens_id, start_ind_id, rng_sens_num_id, start_tm_id, n_stp_id, accels, structured_array):
    S = []
    S_norm = []
    len = 0

    for j, tm in enumerate(start_tm_id):
        for i, start_index in enumerate(start_ind_id):
            sens_len = rng_sens_num_id[i]
            n_steps = n_stp_id[i]
            S_this_track = 0
            if sens_len > 0:
                print(tm, j)
                for k in range(len,len+sens_len):
                    sens_num = sens_id[k]
                    expected_signal = expected_signal_from_sensor[k]
                    signal_from_sensor = accels[sens_num][
                                         start_index:start_index + expected_signal.shape[0]]

                    S_this_track += np.einsum(
                        'ij,ij->', expected_signal, signal_from_sensor
                    )

                    if np.any(np.isnan(S_this_track)): import pdb; pdb.set_trace()
                len += sens_len
                S.append(S_this_track)
                S_norm.append(S_this_track / n_steps)
            else:
                S.append(0)
                S_norm.append(0)

    structured_array['S'] = S
    structured_array['S_norm'] = S_norm

    return structured_array
