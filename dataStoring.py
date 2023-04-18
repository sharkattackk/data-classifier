import pandas as pd
import numpy as np
import h5py as h5
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cleanup
import matplotlib.pyplot as plt

# read CSV files into pandas dataframes
df1 = pd.read_csv('./data/lydia_jumping.csv')
df2 = pd.read_csv('./data/lydia_walking.csv')
df3 = pd.read_csv('./data/cam_jumping.csv')
df4 = pd.read_csv('./data/cam_walking.csv')
df5 = pd.read_csv('./data/msendoo_jumping.csv')
df6 = pd.read_csv('./data/msendoo_walking.csv')
#combine the data
framesJ = [df1, df3, df5]
combinedJ = pd.concat(framesJ)
framesW = [df2, df4, df6]
combinedW = pd.concat(framesW)
cleanJump, cleanWalk = cleanup.cleanThemUp(combinedJ, combinedW)



window_size = 5*100 #100 HZ sample rate
windowsJ=[] #empty array for storing the Jumping windows
for i in range(0, len(cleanJump)-window_size, window_size):
    windowJ = cleanJump[i:i+window_size]
    windowsJ.append(windowJ) #store current window in array
np.random.shuffle(windowsJ) #shuffle the windows
train_dataJ, test_dataJ = train_test_split(windowsJ, test_size=0.1) #split the jumping data


windowsW=[] #empty array for storing the Walking windows
for j in range(0, len(cleanWalk) - window_size, window_size): 
    windowW = cleanWalk[j:j+window_size] 
    windowsW.append(windowW) #store the current window in array
np.random.shuffle(windowsW) #shuffle the windows
train_dataW, test_dataW = train_test_split(windowsW, test_size=0.1) #split the walking data

with h5.File('./h5/project_data.h5', mode='w') as store:
    G4 = store.create_group('/dataset/train')
    G4.create_dataset('cleanJump', data=train_dataJ)
    G4.create_dataset('cleanWalk', data=train_dataW)
    G5 = store.create_group('/dataset/test')
    G5.create_dataset('cleanJump', data=test_dataJ)
    G5.create_dataset('cleanWalk', data=test_dataW)

#each members datagroups for organization
    #adding Lydia's data group
    G1 = store.create_group('/Lydia')
    G1.create_dataset('jumping', data = df1)
    G1.create_dataset('walking', data = df2)

    #adding Camerons data group 
    G2 = store.create_group('/Cameron')
    G2.create_dataset('jumping', data = df3)
    G2.create_dataset('walking', data = df4)

    #adding Msendoo's data group
    G3 = store.create_group('/Msendoo')
    G3.create_dataset('jumping', data = df5)
    G3.create_dataset('walking', data = df6)




