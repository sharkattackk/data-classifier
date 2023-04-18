import pandas as pd
import numpy as np
import h5py as h5
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from scipy.stats import skew

# define a function to extract features from a windows
def extract_features(window):
    features = []
    for col in window.columns:
        features.append(window[col].mean())
        features.append(window[col].median())
        features.append(window[col].min())
        features.append(window[col].max())
        features.append(window[col].var())
        features.append(window[col].max() - window[col].min())
        features.append(skew(window[col]))
        features.append(window[col].std())
        features.append(np.sqrt(np.mean(window[col]**2)))
        features.append(window[col].kurtosis())
    return features


with h5.File('./h5/project_data.h5', 'r') as f:
    train_dataJ = f['/dataset/train/cleanJump']
    train_dataW = f['/dataset/train/cleanWalk']
    test_dataJ = f['/dataset/test/cleanJump']
    test_dataW = f['/dataset/test/cleanWalk']
    train_jump_feat = []
    for i in range(train_dataJ.shape[0]):
        window = pd.DataFrame(train_dataJ[i])
        window_features = extract_features(window)
        train_jump_feat.append(window_features)
    train_walk_feat = []
    for j in range(train_dataW.shape[0]):
        window = pd.DataFrame(train_dataW[j])
        window_features = extract_features(window)
        train_walk_feat.append(window_features)
    test_jump_feat = []
    for l in range(test_dataJ.shape[0]):
        window = pd.DataFrame(test_dataJ[l])
        window_features = extract_features(window)
        test_jump_feat.append(window_features)
    test_walk_feat = []
    for m in range(test_dataW.shape[0]):
        window = pd.DataFrame(test_dataW[m])
        window_features = extract_features(window)
        test_walk_feat.append(window_features)

with h5.File('./h5/features.h5', 'w') as f:
    G1 = f.create_group('features/train')
    G1.create_dataset('jumping', data=train_jump_feat)
    G1.create_dataset('walking', data=train_walk_feat)

    G2 = f.create_group('features/test')
    G2.create_dataset('jumping', data=test_jump_feat)
    G2.create_dataset('walking', data=test_walk_feat)










