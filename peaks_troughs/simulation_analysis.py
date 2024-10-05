import numpy as np


def load_data():
    return np.load('/data/simulations/res.npz',allow_pickle=True)['arr_0'].item()


def detect_zeros(arr):
    is_pos = ( arr[1:]- arr[:-1]) >= 0
    sign_change = np.logical_xor(is_pos[1:], is_pos[:-1])
    return np.argwhere(sign_change)+1
    
    
def track_simulation():
    dic = np.load('/data/simulations/res.npz',allow_pickle=True)['arr_0'].item()