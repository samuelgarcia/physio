'''
Respiration example
===================


'''

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pprint

import physio

##############################################################################
# 
# physio.respiration
# 
# ------------------
#
# 
#  

##############################################################################
# 
# Respiration : quick way
# 
# -----------------------
#
#  The fastest way is to use compute_respiration() using predefine parameters set
#  here a simple example


# read data
raw_resp = np.load('resp1.npy')
srate = 1000.
times = np.arange(raw_resp.size) / srate

# the easiest way is to use predefined parameters
resp, cycle_features = physio.compute_respiration(raw_resp, srate, parameter_set='human_airflow')

# cycle_features is a dataframe containing all cycles and related fetaures (duration, amplitude, volume, timing)
print(cycle_features)

inspi_index = cycle_features['inspi_index'].values
expi_index = cycle_features['expi_index'].values

fig, ax = plt.subplots()
ax.plot(times, raw_resp)
ax.plot(times, resp)
ax.scatter(times[inspi_index], resp[inspi_index], marker='o', color='green')
ax.scatter(times[expi_index], resp[expi_index], marker='o', color='red')
ax.set_ylabel('resp')

ax.set_xlim(350, 450)


##############################################################################
# 
# Parameters tuning
# -----------------------
# 
# Here a simple recipe to change some predefined parameters
# We change here the length smoothing parameter

# get paramseters set
# this is a nested dict of parameter of every step
parameters = physio.get_respiration_parameters('human_airflow')
# lets change on parameter in the structure
parameters['smooth']['sigma_ms'] = 50.
pprint(parameters)

resp, cycle_features = physio.compute_respiration(raw_resp, srate, parameters=parameters)



##############################################################################
# 
# Respiration : step by step
# --------------------------
# 
# Here a detail of all low level functions that are used internally in the compute_respiration()


resp = physio.preprocess(resp, srate, band=25., btype='lowpass', ftype='bessel', order=5, normalize=False)
resp = physio.smooth_signal(resp, srate, win_shape='gaussian', sigma_ms=90.0)

baseline = physio.get_respiration_baseline(resp, srate, baseline_mode='median - epsilon')
print('baseline', baseline)

# this will give a numpy.array with shape (num_cycle, 3)
cycles = physio.detect_respiration_cycles(resp, srate, baseline_mode='manual', baseline=baseline)
print(cycles[:10])

# this will return a dataframe with all cycles and fetaure before cleaning
cycle_features = physio.compute_respiration_cycle_features(resp, srate, cycles, baseline=baseline)

# this will remove outliers cycles based on log ratio distribution
cycle_features = physio.clean_respiration_cycles(resp, srate, cycle_features, baseline, low_limit_log_ratio=3)
print(cycle_features.head(10))


inspi_index = cycle_features['inspi_index'].values
expi_index = cycle_features['expi_index'].values

fig, ax = plt.subplots()
ax.plot(times, resp)
ax.scatter(times[inspi_index], resp[inspi_index], marker='o', color='green')
ax.scatter(times[expi_index], resp[expi_index], marker='o', color='red')
ax.set_ylabel('resp')
ax.axhline(baseline, color='Coral')

ax.set_xlim(350, 450)


plt.show()