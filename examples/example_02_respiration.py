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
# Respiration cycle detection. The faster way: using `parameter_preset`.
# ------------------------------------------
#
# The fastest way to process respiration with :py:mod:`physio` is to use :py:func:`~physio.compute_respiration` which is a high-level wrapper function that simplifies respiratory signal analysis.
# To use this function, you must provide:
#    * `raw_resp` : the raw respiratory signal as a NumPy array.
#    * `srate` : the sampling rate of the respiratory signal
#    * `parameter_preset` : a string specifying the type of respiratory data, which determines the set of parameters used for processing. Can be one of: `human_airflow`, `human_co2`, `rat_plethysmo`, or `rat_etisens_belt`.
# When called, :py:func:`~physio.compute_respiration` performs the following:
#    * Preprocesses the respiratory signal (returns a NumPy array: `resp`)
#    * Computes cycle-by-cycle features (returns a pd.DataFrame: `resp_cycles`)
# **Warning:** The orientation of the `raw_resp` trace is important (multiply it by -1 for reversing it if necessary). 
# Inspiration must point downward for `human_airflow` or `human_co2`, 
# because downward deflections are interpreted by :py:func:`~physio.compute_respiration` as inspiration. 
# For `rat_plethysmo` or `rat_etisens_belt`, the inspiration–expiration transition must point upward.
# 
# For this tutorial, we will use an internal file already stored in NumPy format for demonstration purposes.

raw_resp = np.load('resp_airflow1.npy') # load respi
srate = 1000. # our example signals have been recorded at 1000 Hz

times = np.arange(raw_resp.size) / srate # build time vector


# the easiest way is to use predefined parameters
resp, resp_cycles = physio.compute_respiration(raw_resp, srate, parameter_preset='human_airflow')  # set 'human_airflow' as preset because example resp is an airflow from humans

# resp_cycles is a dataframe containing all cycles and related fetaures (duration, amplitude, volume, timing).
print(resp_cycles)

inspi_ind = resp_cycles['inspi_index'].values # get index of inspiration start points
expi_ind = resp_cycles['expi_index'].values # get index of expiration start points

fig, ax = plt.subplots()
ax.plot(times, raw_resp)
ax.plot(times, resp)
ax.scatter(times[inspi_ind], resp[inspi_ind], color='green', label = 'inspiration start')
ax.scatter(times[expi_ind], resp[expi_ind], color='red', label = 'expiration start')
ax.set_xlim(185, 225)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude (AU)')
ax.set_ylim(-1750, -1450)
ax.set_title('Respiratory cycle detection')
ax.legend(loc = 'upper right')

ax.set_xlim(110, 170)

##############################################################################
# 
# What is `parameter_preset` ?
# ---------------------------
# 
# Using `parameter_preset` tells :py:func:`~physio.compute_respiration` to process respiration
# according to a predefined set of parameters already optimized by :py:mod:`physio`.
# 
# To get an idea of the default parameters used in `parameter_preset`, 
# you can call :py:func:`~physio.get_respiration_parameters`.
# 
# Here is an example showing the set of parameters applied in the case of a human airflow signal.


parameters = physio.get_respiration_parameters('human_airflow') # parameters is a nested dictionary of parameters used at each processing step.
pprint(parameters) # pprint to "pretty print"

##############################################################################
# 
# Tuning parameters if unsatisfied
# ---------------------------------
# 
# Variability during data acquisition (subject, acquisition system) can affect the recorded respiratory signal.
# Such variability may make some predefined parameters of :py:mod:`physio` inappropriate. 
# 
# In this situation, you can tune certain parameters by re-assigning values to the keys of the `parameters` dictionary.
# You may also tune multiple parameters at once if necessary. 
# To fine-tune parameters properly, a good understanding of each parameter's role is required. 
# For this reason, we have dedicated a whole section to this topic — see the "Parameters" section.
# 
# For example, here we modify the length of the smoothing parameter, which corresponds to the duration in milliseconds
# of the Gaussian smoothing kernel (the width of the bell-shaped curve), usually referred to as sigma.




# let's change one parameter in the nested structure ...
parameters['smooth']['sigma_ms'] = 100. # at the key "smooth" and the sub-key "sigma_ms", the default values is 60. Here we replace it by 100 milliseconds to induce more smoothing
pprint(parameters) # pprint to "pretty print" 

# ... and use them by providing it to "parameters"
resp, resp_cycles = physio.compute_respiration(raw_resp, srate, parameters=parameters) # preset_parameters = None in this case, because parameters is now explicitly defined



##############################################################################
# 
# Using low-level functions to control each step of the pipeline
# --------------------------------------------------------------
# 
# :py:func:`~physio.compute_respiration` is a high-level wrapper function that simplifies respiratory signal analysis. 
# However, you may want more control over the entire process and therefore use the low-level functions of :py:mod:`physio`.
# Here are the details of all low-level functions used internally by :py:func:`~physio.compute_respiration`.



resp = physio.preprocess(raw_resp, srate, band=25., btype='lowpass', ftype='bessel', order=5, normalize=False) # filter
resp = physio.smooth_signal(resp, srate, win_shape='gaussian', sigma_ms=90.0) # smooth

baseline = physio.get_respiration_baseline(resp, srate, baseline_mode='median') # compute baseline level
print('baseline :', baseline)

# this will give a numpy.array with shape (num_cycle, 3)
cycles = physio.detect_respiration_cycles(resp, srate, baseline_mode='manual', baseline=baseline) # detect inspi-expi / expi-inspi / next inspi-expi indices
print(cycles[:10])

# this will return a dataframe with all cycles and features before cleaning
resp_cycles = physio.compute_respiration_cycle_features(resp, srate, cycles, baseline=baseline) # compute cycle-by-cycle resp features

# this will remove outliers cycles based on log ratio distribution
resp_cycles = physio.clean_respiration_cycles(resp, srate, resp_cycles, baseline, low_limit_log_ratio=3) # clean features
print(resp_cycles.head(10))


inspi_ind = resp_cycles['inspi_index'].values # get index of inspiration start points
expi_ind = resp_cycles['expi_index'].values # get index of expiration start points

fig, ax = plt.subplots()
ax.plot(times, resp, label = 'preprocessed resp signal')
ax.scatter(times[inspi_ind], resp[inspi_ind], marker='o', color='green')
ax.scatter(times[expi_ind], resp[expi_ind], marker='o', color='red')
ax.axhline(baseline, color='Coral', label = 'baseline', ls = '--', alpha = 0.9)
ax.set_ylabel('Amplitude (AU)')
ax.set_xlabel('Time (s)')
ax.set_title('Respiratory cycle detection (using low-level functions)')
ax.set_ylim(-1750, -1450)
ax.set_xlim(110, 170)
ax.legend(loc = 'upper right')

plt.show()
