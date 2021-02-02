'''integral transform module.'''
import json
from numba import njit
import numpy as np
from scipy import signal
from tqdm import tqdm, trange
from planckanalysis.separated_integral_transform import signal_function

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
