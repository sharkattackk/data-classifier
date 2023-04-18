import pandas as pd
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
def cleanThemUp(dataOne, dataTwo):
    window_size = 31
    cols = ['Time (s)', 'Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)', 'Linear Acceleration z (m/s^2)', 'Absolute acceleration (m/s^2)' ]
    jump_data = dataOne[cols]
    cleanOne = jump_data.rolling(window_size).mean().dropna()
    cleanOne.interpolate(method='linear', inplace=True)
    walk_data = dataTwo[cols]
    cleanTwo = walk_data.rolling(window_size).mean().dropna()
    cleanTwo.interpolate(method='linear', inplace=True)
    return cleanOne, cleanTwo
def cleanItUp(data):
    window_size = 31
    cols = [0, 1, 2, 3, 4]
    jump_data = data[cols]
    clean = jump_data.rolling(window_size).mean().dropna()
    clean.interpolate(method='linear', inplace=True)
    return clean
def clean(data):
    window_size = 31
    cols = ['Time (s)', 'Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)', 'Linear Acceleration z (m/s^2)', 'Absolute acceleration (m/s^2)' ]
    jump_data = data[cols]
    clean = jump_data.rolling(window_size).mean().dropna()
    clean.interpolate(method='linear', inplace=True)
    return clean