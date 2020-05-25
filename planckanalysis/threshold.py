import numpy as np

def threshold(parameter_list, min_signal=0):
    '''Takes signal values as well as entry and exit 4-vector lists as 
    arguments. We throw out all tracks that do not have a signal value
    above some minimum (to be determined later). We put all aformentioned 
    values into a structured numpy array return that structured array.'''

    rows_to_keep = []

    for y,S in enumerate(parameter_list['S']):
        if S > min_signal:
            rows_to_keep.append(y)


    return parameter_list[rows_to_keep]