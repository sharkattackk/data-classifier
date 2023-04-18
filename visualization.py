import h5py as h5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read CSV files into pandas dataframes
df1 = pd.read_csv('./data/lydia_jumping.csv')
df2 = pd.read_csv('./data/lydia_walking.csv')
df3 = pd.read_csv('./data/cam_jumping.csv')
df4 = pd.read_csv('./data/cam_walking.csv')
df5 = pd.read_csv('./data/msendoo_jumping.csv')
df6 = pd.read_csv('./data/msendoo_walking.csv')

#combined 
framesJ = [df1, df3, df5]
jumping_data = pd.concat(framesJ, keys=["x", "y", "z"])
framesW = [df2, df4, df6]
walking_data = pd.concat(framesW, keys=["x", "y", "z"])

# Plot acceleration vs. time for Jumping Data
fig, axJ = plt.subplots(nrows=2, ncols= 2, figsize=(10, 10))
axJ[0,0].plot(jumping_data['Time (s)'], jumping_data['Absolute acceleration (m/s^2)'])
axJ[0,0].set_xlabel('Time (s)')
axJ[0,0].set_ylabel('Absolute Acceleration (m/s^2)')
axJ[0,0].set_title('Absolute Acceleration vs time while jumping')
#plot for acceleration vs time on the X axis
axJ[0,1].plot(jumping_data['Time (s)'], jumping_data['Linear Acceleration x (m/s^2)'])
axJ[0,1].set_xlabel('Time (s)')
axJ[0,1].set_ylabel('Linear Acceleration X axis (m/s^2)')
axJ[0,1].set_title('Linear Acceleration X axis vs time while jumping')
#plot for acceleration vs time on the Y axis
axJ[1,0].plot(jumping_data['Time (s)'], jumping_data['Linear Acceleration y (m/s^2)'])
axJ[1,0].set_xlabel('Time (s)')
axJ[1,0].set_ylabel('Linear Acceleration Y axis (m/s^2)')
axJ[1,0].set_title('Linear Acceleration Y axis vs time while jumping')
#plot for acceleration vs time on the Z axis
axJ[1,1].plot(jumping_data['Time (s)'], jumping_data['Linear Acceleration z (m/s^2)'])
axJ[1,1].set_xlabel('Time (s)')
axJ[1,1].set_ylabel('Linear Acceleration Z axis (m/s^2)')
axJ[1,1].set_title('Linear Acceleration Z axis vs time while jumping')


# Plot acceleration vs. time for walking Data
fig, axW = plt.subplots(nrows=2, ncols= 2, figsize=(10, 10))
axW[0,0].plot(walking_data['Time (s)'], walking_data['Absolute acceleration (m/s^2)'])
axW[0,0].set_xlabel('Time (s)')
axW[0,0].set_ylabel('Absolute Acceleration (m/s^2)')
axW[0,0].set_title('Absolute Acceleration vs time while walking')
#plot for acceleration vs time on the X axis
axW[0,1].plot(walking_data['Time (s)'], walking_data['Linear Acceleration x (m/s^2)'])
axW[0,1].set_xlabel('Time (s)')
axW[0,1].set_ylabel('Linear Acceleration X axis (m/s^2)')
axW[0,1].set_title('Linear Acceleration X axis vs time while walking')
#plot for acceleration vs time on the Y axis
axW[1,0].plot(walking_data['Time (s)'], walking_data['Linear Acceleration y (m/s^2)'])
axW[1,0].set_xlabel('Time (s)')
axW[1,0].set_ylabel('Linear Acceleration Y axis (m/s^2)')
axW[1,0].set_title('Linear Acceleration Y axis vs time while walking')
#plot for acceleration vs time on the Z axis
axW[1,1].plot(walking_data['Time (s)'], walking_data['Linear Acceleration z (m/s^2)'])
axW[1,1].set_xlabel('Time (s)')
axW[1,1].set_ylabel('Linear Acceleration Z axis (m/s^2)')
axW[1,1].set_title('Linear Acceleration Z axis vs time while walking')

plt.show()
