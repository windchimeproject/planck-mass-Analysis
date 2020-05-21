def threshold(signals, entry_list, exit_list):
    '''Takes signal values as well as entry and exit 4-vector lists as 
    arguments. We throw out all tracks that do not have a signal value
    above some minimum (to be determined later). We put all aformentioned 
    values into a structured numpy array return that structured array.'''
