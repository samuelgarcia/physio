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
# ----------------------------------------------------------------------
#
# The fastest way to process respiration with :py:mod:`physio` is to use :py:func:`~physio.compute_respiration` which is a high-level wrapper function that simplifies respiratory signal analysis.
# To use this function, you must provide:
#    * `raw_resp` : the raw respiratory signal as a NumPy array.
#    * `srate` : the sampling rate of the respiratory signal
#    * `parameter_preset` : a string specifying the type of respiratory data, which determines the set of parameters used for processing. Can be one of: `human_airflow`, `human_co2`, `human_belt`, `rat_plethysmo`, or `rat_etisens_belt`.
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

# resp_cycles is a dataframe containing all cycles and related features (duration, amplitude, volume, timing, etc...).
print(resp_cycles)

inspi_ind = resp_cycles['inspi_index'].values # get index of inspiration start points
expi_ind = resp_cycles['expi_index'].values # get index of expiration start points

fig, ax = plt.subplots()
ax.plot(times, raw_resp)
ax.plot(times, resp)
ax.scatter(times[inspi_ind], resp[inspi_ind], color='green', label = 'inspiration start')
ax.scatter(times[expi_ind], resp[expi_ind], color='red', label = 'expiration start')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude (AU)')
ax.set_ylim(-1750, -1450)
ax.set_title('Respiratory cycle detection')
ax.legend(loc = 'upper right')

ax.set_xlim(110, 170)

##############################################################################
# 
# What is `parameter_preset` ?
# ----------------------------
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
# --------------------------------
# 
# Variability during data acquisition (subject, acquisition system) can affect the recorded respiratory signal.
# Such variability may make some predefined parameters of :py:mod:`physio` inappropriate. 
# 
# In this situation, you can tune certain parameters by re-assigning values to the keys of the `parameters` dictionary.
# You may also tune multiple parameters at once if necessary. 
# **To fine-tune parameters properly, a good understanding of each parameter's role is required. 
# For this reason, we have dedicated a whole section to this topic — see the "Parameters" section.**
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
resp_cycles = physio.compute_respiration_cycle_features(resp, srate, cycles, baseline=baseline, sensor_type = 'airflow') # compute cycle-by-cycle resp features on airflow sensor type

# this will remove outliers cycles based on log ratio distribution
resp_cycles = physio.clean_respiration_cycles(resp, srate, resp_cycles, baseline, low_limit_log_ratio=4.5, sensor_type = 'airflow') # clean features
print(resp_cycles.head(10))


inspi_ind = resp_cycles['inspi_index'].values # get index of inspiration start points
expi_ind = resp_cycles['expi_index'].values # get index of expiration start points

fig, ax = plt.subplots()
ax.plot(times, resp, label = 'preprocessed resp signal')
ax.scatter(times[inspi_ind], resp[inspi_ind], marker='o', color='green', label = 'inspiration start')
ax.scatter(times[expi_ind], resp[expi_ind], marker='o', color='red', label = 'inspiration start')
ax.axhline(baseline, color='Coral', label = 'baseline', ls = '--', alpha = 0.9)
ax.set_ylabel('Amplitude (AU)')
ax.set_xlabel('Time (s)')
ax.set_title('Respiratory cycle detection (using low-level functions)')
ax.set_ylim(-1750, -1450)
ax.set_xlim(110, 170)
ax.legend(loc = 'upper right')

plt.show()


##############################################################################
# 
# Respiration features / metrics
# ------------------------------
# 
# `resp_cycles` is a dataframe containing one row per respiratory cycle and one
# column per feature. Depending on the sensor type, **each cycle is described by
# multiple features.**
# 
# Some features are related to the position (index) or time of particular points
# within the cycle:
# 
#    * `inspi_index`: Index of the start of inspiration (green point in the figure below).
#      The index refers to the sequential position of the sample in the entire respiratory
#      time series.
#    * `expi_index`: Index of the start of expiration (red point in the figure below).
#    * `next_inspi_index`: Index of the start of the next inspiration (equal to the
#      `inspi_index` of cycle n+1).
#    * `inspi_time`: (units = `s`) Time of the start of inspiration (green point in the figure below).
#    * `expi_time`: (units = `s`) Time of the start of expiration (red point in the figure below).
#    * `next_inspi_time`: (units = `s`) Time of the start of the next inspiration.
#    * `inspi_peak_index`: Index of the inspiratory peak = position of the minimum
#      sample of the cycle in the respiratory time series (computed only if
#      `sensor_type` = `airflow`).
#    * `expi_peak_index`: Index of the expiratory peak = position of the maximum
#      sample of the cycle in the respiratory time series (computed only if
#      `sensor_type` = `airflow`).
#    * `inspi_peak_time`: (units = `s`) Time of the inspiratory peak = timestamp of the minimum
#      sample of the cycle (computed only if `sensor_type` = `airflow`).
#    * `expi_peak_time`: (units = `s`) Time of the expiratory peak = timestamp of the maximum
#      sample of the cycle (computed only if `sensor_type` = `airflow`).
# 
# From these points, many derived features of interest (e.g., for statistics) are computed:
# 
#    * `cycle_duration`: (units = `s`) Duration of the cycle in seconds = `next_inspi_time` - `inspi_time`
#    * `inspi_duration`: (units = `s`) Duration of inspiration in seconds = `expi_time` - `inspi_time`
#    * `expi_duration`: (units = `s`) Duration of expiration in seconds = `next_inspi_time` - `expi_time`
#    * `cycle_freq`: (units = `Hz`) Breathing frequency in Hertz = 1 / `cycle_duration`
#    * `cycle_ratio`: (units = AU because proportion) Ratio of inspiration duration to total cycle duration
#      = `inspi_duration` / `cycle_duration`. Equivalent to the relative position
#      of the transition from inspiration to expiration.
#    * `inspi_amplitude`: (units = AU, the units of the recorded respiratory signal) Amplitude difference of the respiratory signal from baseline
#      at `inspi_peak_index` (computed only if `sensor_type` = `airflow` or `belt`).
#      Equivalent to peak inspiratory flow (black descending arrow in the figure below).
#    * `expi_amplitude`: (units = AU, the units of the recorded respiratory signal) Amplitude difference of the respiratory signal from baseline
#      at `expi_peak_index` (computed only if `sensor_type` = `airflow`' or `belt`).
#      Equivalent to peak expiratory flow (black ascending arrow in the figure below).
#    * `total_amplitude`: (units = AU, the units of the recorded respiratory signal) Sum of `inspi_amplitude` + `expi_amplitude` (computed only if
#      `sensor_type` = `airflow` or `belt`).
#    * `inspi_volume`: (units = AU * `s`, depending on units the recorded respiratory signal) Integral of the respiratory signal below baseline during
#      inspiration (computed only if `sensor_type` = `airflow`). Equivalent to the
#      green area in the figure below.
#    * `expi_volume`: (units = AU * `s`, depending on units the recorded respiratory signal) Integral of the respiratory signal above baseline during
#      expiration (computed only if `sensor_type` = `airflow`). Equivalent to the
#      red area in the figure below.
#    * `total_volume`: (units = AU * `s`, depending on units the recorded respiratory signal) Sum of `inspi_volume` + `expi_volume` (computed only if
#      `sensor_type` = `airflow`). Equivalent to the sum of the green + red areas
#      in the figure below.
#
# .. image:: ./img/resp_features_doc_physio.png
#    :alt: Respiration Parameters
#    :align: center
#    :scale: 70%

pprint(f'{resp_cycles.shape[0]} detected cycles')
pprint(f'{resp_cycles.shape[1]} computed features, which are the following :')
pprint(resp_cycles.columns.to_list())


##############################################################################
# 
# Quick example with respiration recorded using a belt
# ----------------------------------------------------
# 
# For this tutorial, we will use an internal file already stored in NumPy 
# format for demonstration purposes. This file corresponds to 5 minutes of 
# signal recorded with a belt on a human subject. The belt is a sensor of trunk 
# circumference: the signal rises when the trunk dilates and decreases when the 
# trunk retracts.
# 
# Therefore, the detection method does not rely on a "baseline-crossing" 
# approach, but rather on a "min-max" approach. This `min_max` method is 
# activated via the `parameter_preset` dedicated to this situation.
# 
# Note that respiratory signals recorded with belts are often of poor quality. 
# In addition, some participants may present paradoxical respiration, with the 
# trunk retracting during inspiration and dilating during expiration. In such 
# cases, the metrics returned in `resp_cycles_belt` will not be meaningful.
# 
# Let's run a short example.
# 



raw_resp_belt = np.load('resp_belt3.npy') # load respi belt
srate = 1000. # our example signals have been recorded at 1000 Hz

times = np.arange(raw_resp_belt.size) / srate # build time vector


# the easiest way is to use predefined parameters
resp_belt, resp_cycles_belt = physio.compute_respiration(raw_resp_belt, srate, parameter_preset='human_belt')  # set 'human_belt' as preset because example resp is an airflow from humans

# print human_belt params
pprint(physio.get_respiration_parameters('human_belt'))

# resp_cycles_belt is a dataframe containing all cycles and related features (duration, amplitude, timing, etc...). In the case of belt, volumes are not computed.
print(resp_cycles_belt)

inspi_ind = resp_cycles_belt['inspi_index'].values # get index of inspiration start points
expi_ind = resp_cycles_belt['expi_index'].values # get index of expiration start points

fig, ax = plt.subplots()
ax.plot(times, raw_resp_belt)
ax.plot(times, resp_belt)
ax.scatter(times[inspi_ind], resp_belt[inspi_ind], color='green', label = 'inspiration start')
ax.scatter(times[expi_ind], resp_belt[expi_ind], color='red', label = 'expiration start')
ax.set_xlim(195, 215)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude (AU)')
ax.set_title('Respiratory cycle detection on a belt signal (min-max signal)')
ax.legend(loc = 'upper right')



##############################################################################
#
# Quick example with respiration recorded using a CO2 sensor
# ----------------------------------------------------------
#
# For this tutorial, we will use an internal file already stored in NumPy 
# format for demonstration purposes. This file corresponds to 5 minutes of 
# signal recorded with a CO2 sensor at 60 Hz from a human subject. 
#
# In the upper airways, CO2 concentration decreases during inspiration 
# and increases during exhalation, and so does the recorded signal.
#
# Therefore, the detection method does not rely on a "baseline-crossing" 
# approach, but rather on a dedicated "co2" method. This method is 
# activated via the `parameter_preset` specific to this situation.
#
# Let's run a short example.
#




raw_resp_co2 = np.load('resp_CO2_4.npy') # load respi co2
srate = 60. # our example signals have been recorded at 60 Hz

times = np.arange(raw_resp_co2.size) / srate # build time vector


# the easiest way is to use predefined parameters
resp_co2, resp_cycles_co2 = physio.compute_respiration(raw_resp_co2, srate, parameter_preset='human_co2')  # set 'human_co2' as preset because example resp is an airflow from humans

# print human_co2 params
pprint(physio.get_respiration_parameters('human_co2'))

# resp_cycles_co2 is a dataframe containing all cycles and related features (duration, timing, etc...). In the case of belt, volumes and amplitudes are not computed.
print(resp_cycles_co2)

inspi_ind = resp_cycles_co2['inspi_index'].values # get index of inspiration start points
expi_ind = resp_cycles_co2['expi_index'].values # get index of expiration start points

fig, ax = plt.subplots()
ax.plot(times, raw_resp_co2)
ax.plot(times, resp_co2)
ax.scatter(times[inspi_ind], resp_co2[inspi_ind], color='green', label = 'inspiration start')
ax.scatter(times[expi_ind], resp_co2[expi_ind], color='red', label = 'expiration start')
ax.set_xlim(95, 110)
ax.set_ylim(-1, 40)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude (mmHg)')
ax.set_title('Respiratory cycle detection on a CO2 signal')
ax.legend(loc = 'upper right')

