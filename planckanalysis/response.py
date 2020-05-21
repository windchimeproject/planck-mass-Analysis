'''detector response module.'''
import json

import numpy as np
from scipy import signal

from .config import CONFIG
from .detector_characteristics import DETECTOR_CHARACTERISTICS

RESPONSE_FILE = CONFIG['Detector Config']['ResponseFile']

with open(RESPONSE_FILE) as f:
    RESPONSE_DICT = json.load(f)

def sensor_response(sensor, acceleration, response_dict=RESPONSE_DICT):
    '''returns ADC value based on true MC acceleration'''

    linear_response = response_dict[sensor]['linear_response']
    sensitivity = DETECTOR_CHARACTERISTICS[sensor]['sensitivity']
    #scaling = 2.0*np.pi
    noise = DETECTOR_CHARACTERISTICS[sensor]['noise']*np.random.randn(*acceleration.shape)
    acceleration_w_noise = acceleration + noise
    convolved_list = []
    for dim in range(acceleration_w_noise.shape[1]):
        voltage = acceleration_w_noise[:, dim]*sensitivity[dim]
        convolved_list.append(signal.convolve(voltage, linear_response))
    convolved_signal = np.array(convolved_list).T

    signal_transfer_response = response_dict[sensor]['signal_transfer_response']
    output_signal = np.searchsorted(signal_transfer_response, convolved_signal)

    return output_signal
