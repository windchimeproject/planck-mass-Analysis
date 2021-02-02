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


def Non_Spatial_Analysis_alphas(vel, entry_vecs, exit_vecs):
    alphas = np.zeros((len(vel), 8))
    for i, vel_p in enumerate(tqdm(vel)):
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


def Spatial_Analysis_alphas(vel, radius, entry_Anl=False, exit_Anl=False, entry_vals=[-1, -1], exit_vals=[-1, -1],
                            N_thetas=15, N_phis_at_eq=30, epsilon=0):
    if exit_Anl == True:
        theta_start = entry_vals[0] * 180 / np.pi
    else:
        theta_start = np.linspace(0, 180, N_thetas)

    if entry_Anl == True:
        theta_end = exit_vals[0] * 180 / np.pi
    else:
        theta_end = np.linspace(0, 180, N_thetas)

    angles = []
    alphas = []

    if entry_Anl == True and any(val < 0 for val in exit_vals) == True:
        raise ValueError(
            "You have indicated an entry spatial analysis but have not provided the truth exit theta and phi values. Please input with the form: exit_vals=[theta_exit_truth, phi_exit_truth].")
    if exit_Anl == True and any(val < 0 for val in entry_vals) == True:
        raise ValueError(
            "You have indicated an exit spatial analysis but have not provided the truth entry theta and phi values. Please input with the form: entry_vals=[theta_entry_truth, phi_entry_truth].")

    for vel_p in tqdm(vel):
        for theta_entry in theta_start:
            for theta_exit in theta_end:
                if theta_exit > theta_entry + epsilon or theta_exit < theta_entry - epsilon:
                    # The phi calcs below are dependent on the particular thetas considered to keep the spherical uniformity of the spatial analysis consideration
                    if exit_Anl == True:
                        phi_start = [entry_vals[1] * 180 / np.pi]
                    else:
                        phi_start = np.linspace(-180, 180,
                                                int(np.round(N_phis_at_eq * np.cos((theta_entry - 90) * np.pi / 180))))

                    if len(phi_start) == 0:
                        phi_start = [0]

                    for phi_entry in phi_start:
                        if entry_Anl == True:
                            phi_end = [exit_vals[1] * 180 / np.pi]
                        else:
                            phi_end = np.linspace(-180, 180, int(
                                np.round(N_phis_at_eq * np.cos((theta_entry - 90) * np.pi / 180))))

                        if len(phi_end) == 0:
                            phi_end = [0]

                        for phi_exit in phi_end:
                            if phi_exit > phi_entry + epsilon or phi_exit < phi_entry - epsilon:
                                x_0 = radius * np.sin(theta_entry * np.pi / 180) * np.cos(phi_entry * np.pi / 180)
                                y_0 = radius * np.sin(theta_entry * np.pi / 180) * np.sin(phi_entry * np.pi / 180)
                                z_0 = radius * np.cos(theta_entry * np.pi / 180)

                                x_1 = radius * np.sin(theta_exit * np.pi / 180) * np.cos(phi_exit * np.pi / 180)
                                y_1 = radius * np.sin(theta_exit * np.pi / 180) * np.sin(phi_exit * np.pi / 180)
                                z_1 = radius * np.cos(theta_exit * np.pi / 180)

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
                                    length / vel_p
                                ])

                                angles.append([
                                    theta_entry * np.pi / 180,
                                    theta_exit * np.pi / 180,
                                    phi_entry * np.pi / 180,
                                    phi_exit * np.pi / 180
                                ])
    return np.array(alphas), np.array(angles)


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


def transform(times, accels, start_times, start_time_indices, alphas, sensors_pos, lin_resp):
    '''Takes time series data as an input and generates a signal value based on
    entry and exit 4-vectors on a sphere extended in time. Returns the signal
    value and 4-vectors. Refer to Qin's note for a much more detailed
    explanation.

    accels is a list of accelerations.

    sensor_dict is a list of sensor positions, in the same order as accels.
    '''

    '''The output of this conjoined integral transform function is the S value, not the SNR.
    And this value cannot be transformed into the SNR trivially. This is because of how the
    function is set up and how the expected signal from the sensor is generated. Here, for
    every sensor, an expected signal is generated, analyzed with the data, then discarded.
    The result of each run is added continously and the final value is the S, but each
    expected signal has been lost and cannot be used for the SNR calculation (see the
    separated integral transform for comparison).'''

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
    steps = []
    adc_timestep_size = times[1] - times[0]
    response_length = len(lin_resp)
    for i, start_time in enumerate(tqdm(start_times)):
        for alpha_index in range(alphas.shape[0]):
            alpha_pair = alphas[alpha_index, :]
            start_index = start_time_indices[i]
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

            S_this_track = 0

            if n_steps > 0:
                for sens_num, sensor_pos in enumerate(sensors_pos):
                    vector_delta = np.zeros((n_steps, 4))
                    for j in range(n_steps):
                        vector_delta[j, 0] = (particle_pos_arr[j][0] - sensor_pos[0])
                        vector_delta[j, 1] = (particle_pos_arr[j][1] - sensor_pos[1])
                        vector_delta[j, 2] = (particle_pos_arr[j][2] - sensor_pos[2])
                        ##vector_delta[j, 3] = (track_times[j] - #step_value) # Something like this can be used for analysis of response time

                    expected_signal_from_sensor = signal_function(vector_delta, lin_resp, adc_timestep_size)
                    signal_from_sensor = accels[sens_num][
                                         start_index:start_index + expected_signal_from_sensor.shape[0]]

                    S_this_track += np.einsum(
                        'ij,ij->', expected_signal_from_sensor, signal_from_sensor
                    )

                    if np.any(np.isnan(S_this_track)): import pdb; pdb.set_trace()

                S.append(S_this_track)
                S_norm.append(S_this_track / n_steps)
            else:
                S.append(0)
                S_norm.append(0)
            alpha0_x.append(alpha_pair[0])
            alpha0_y.append(alpha_pair[1])
            alpha0_z.append(alpha_pair[2])
            alpha0_t.append(alpha_pair[3] + start_time)
            alpha1_x.append(alpha_pair[4])
            alpha1_y.append(alpha_pair[5])
            alpha1_z.append(alpha_pair[6])
            alpha1_t.append(alpha_pair[7] + start_time)
            steps.append(n_steps)
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
        ('steps', 'i4'),
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
    structured_array['steps'] = steps

    return structured_array