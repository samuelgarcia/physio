'''
Respiration tutorial
====================


'''

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pprint

import physio



##############################################################################
# 
# Respiration cycle detection: the quick way
# ------------------------------------------
#
# The fastest way is to use compute_respiration() using a predefined
# parameter preset. Here is a simple example.


# read data
raw_resp = np.load('resp_airflow1.npy')
srate = 1000.
times = np.arange(raw_resp.size) / srate

# the easiest way is to use predefined parameters
resp, resp_cycles = physio.compute_respiration(raw_resp, srate, parameter_preset='human_airflow')

# resp_cycles is a dataframe containing all cycles and related fetaures (duration, amplitude, volume, timing)
print(resp_cycles)

inspi_index = resp_cycles['inspi_index'].values
expi_index = resp_cycles['expi_index'].values

fig, ax = plt.subplots()
ax.plot(times, raw_resp)
ax.plot(times, resp)
ax.scatter(times[inspi_index], resp[inspi_index], marker='o', color='green')
ax.scatter(times[expi_index], resp[expi_index], marker='o', color='red')
ax.set_ylabel('resp')

ax.set_xlim(110, 170)

##############################################################################
# 
# Cycle detection: Default parameters
# -----------------------------------
# 
# Here are the default parameters : those aiming to process a human airflow signal. 
# It is possible to get default parameters dictionnary by calling it with the :py:func:`~physio.get_respiration_parameters` function.

# this is a nested dict of parameters of every processing step
parameters = physio.get_respiration_parameters('human_airflow')
pprint(parameters)

##############################################################################
# 
# Cycle detection: Parameters tuning
# -----------------------------------
# 
# If necessary, it is possible to change the predefined parameters.
# For example, we change here the length of the smoothing parameter.

# let's change on parameter in the structure ...
parameters['smooth']['sigma_ms'] = 100.
pprint(parameters)

# ... and use them by providing it to "parameters"
resp, resp_cycles = physio.compute_respiration(raw_resp, srate, parameters=parameters)



##############################################################################
# 
# Respiration : step by step
# --------------------------
# 
# Here are details of all low-level functions used internally in the compute_respiration()


resp = physio.preprocess(resp, srate, band=25., btype='lowpass', ftype='bessel', order=5, normalize=False)
resp = physio.smooth_signal(resp, srate, win_shape='gaussian', sigma_ms=90.0)

baseline = physio.get_respiration_baseline(resp, srate, baseline_mode='median')
print('baseline', baseline)

# this will give a numpy.array with shape (num_cycle, 3)
cycles = physio.detect_respiration_cycles(resp, srate, baseline_mode='manual', baseline=baseline)
print(cycles[:10])

# this will return a dataframe with all cycles and fetaure before cleaning
resp_cycles = physio.compute_respiration_cycle_features(resp, srate, cycles, baseline=baseline)

# this will remove outliers cycles based on log ratio distribution
resp_cycles = physio.clean_respiration_cycles(resp, srate, resp_cycles, baseline, low_limit_log_ratio=3)
print(resp_cycles.head(10))


inspi_index = resp_cycles['inspi_index'].values
expi_index = resp_cycles['expi_index'].values

fig, ax = plt.subplots()
ax.plot(times, resp)
ax.scatter(times[inspi_index], resp[inspi_index], marker='o', color='green')
ax.scatter(times[expi_index], resp[expi_index], marker='o', color='red')
ax.set_ylabel('resp')
ax.axhline(baseline, color='Coral')

ax.set_xlim(110, 170)


plt.show()
