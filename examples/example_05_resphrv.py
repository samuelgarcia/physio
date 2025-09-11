'''
RespHRV Tutorial
================

Respiratory Heart Rate Variability (RespHRV; previously called Respiratory Sinus Arrhythmia, RSA — see this paper explaining the redefinition of the term: 10.1038/s41569-025-01160-z) can be analyzed using the :py:mod:`physio` toolbox, which provides an innovative method to extract features from heart rate dynamics on a respiratory cycle-by-cycle basis.

The method consists of:
  * Detecting respiratory cycles
  * Detecting ECG peaks
  * Computing instantaneous heart rate in beats per minute
  * Extracting features of this heart rate time series for each respiratory cycle
  * Using cyclic deformation of the heart rate time series and stacking all epochs

This method has two important advantages:
  * RespHRV features can be finely obtained for each respiratory cycle
  * Heart rate dynamics can be analyzed at each phase bin of the respiratory cycle
'''


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from pprint import pprint
from scipy import stats

import physio



##############################################################################
# 
# Read data
# ----------
#  
# For this tutorial, we will use an internal file stored in NumPy format for demonstration purposes.
# See :ref:`sphx_glr_examples_example_01_getting_started.py`, first section, for a description of 
# the capabilities of :py:mod:`physio` for reading raw data formats.


raw_resp = np.load('resp_airflow2.npy') # load respi
raw_ecg = np.load('ecg2.npy') # load ecg
srate = 1000. # our example signals have been recorded at 1000 Hz

times = np.arange(raw_resp.size) / srate # build time vector


##############################################################################
# 
# Get respiratory cycles and ECG peaks using `parameter_preset`
# -------------------------------------------------------------
#  
# See :ref:`sphx_glr_examples_example_02_respiration.py` and 
# :ref:`sphx_glr_examples_example_03_ecg.py` for a detailed explanation of how to use 
# :py:func:`~physio.compute_respiration` and :py:func:`~physio.compute_ecg`, respectively.

resp, resp_cycles = physio.compute_respiration(raw_resp, srate, parameter_preset='human_airflow') # set 'human_airflow' as preset because example resp is an airflow from human
ecg, ecg_peaks = physio.compute_ecg(raw_ecg, srate, parameter_preset='human_ecg') # set 'human_ecg' as preset because example ecg is from human


##############################################################################
# 
# Compute RespHRV
# -----------
#  
# :py:func:`~physio.compute_resphrv` is a high-level wrapper function that computes 
# RespHRV metrics from previously detected R peaks (`ecg_peaks`) and respiratory cycles (`resp_cycles`).
# To use this function, you must provide the previously detected R peaks and respiratory cycles, along with other optional parameters:
#    * `resp_cycles`: `pd.DataFrame`, output of the function :py:func:`~physio.compute_respiration`
#    * `ecg_peaks`: `pd.DataFrame`, output of the function :py:func:`~physio.compute_ecg`
#    * `srate`: `int` or `float`. (optional) Sampling rate used for interpolation to get an instantaneous heart rate vector from RR intervals. 100 Hz is safe for both animal and human.
#    * `units` : `str` (`bpm` / `s` / `ms` / `Hz`), sets the output units (optional, default = 'bpm').
#    * `limits` : `list` or `None`, (optional) range in the chosen units for removing outliers (e.g., [30, 200] to exclude bpm values outside this range). Default is None, meaning no cleaning..
#    * `two_segment` : `bool`, (optional, default = `True`), `True` or `False`, to perform cyclical deformation deviding each respiratory cycle in 2 (if `True`) segments (with the mean `cycle_ratio` of the respiratory cycles as a `segment_ratios`) or 1 (if `False`).  See :ref:`sphx_glr_examples_example_04_cyclic_deformation.py` for more informations.
#    * `points_per_cycle` : `int`, (optional, default = 50), number of points per cycle used for linear resampling during cyclical deformation (see :ref:`sphx_glr_examples_example_04_cyclic_deformation.py`).
#
# When called, :py:func:`~physio.compute_resphrv` performs the following:
#    * Computes instantaneous heart rate (IHR) vector from RR intervals computed from `ecg_peaks`.
#    * Perform cyclic deformation of the IHR vector according to respiratory time points (returns a NumPy array: `cyclic_cardiac_rate` of shape (`n_resp_cycles` * `points_per_cycle`))
#    * Computes Heart-Rate features for each respiratory cycle period (returns a pd.DataFrame array: `resphrv_cycles`)

points_per_cycle = 50 # set number of points per cycle, future number of resp phase points of cyclic_cardiac_rate matrix

resphrv_cycles, cyclic_cardiac_rate = physio.compute_resphrv(
    resp_cycles, # give resp_cycles
     ecg_peaks, # give ecg_peaks
     srate=100., # 100 Hz is safe for both animal and human.
     limits = [30, 200], # 30 to 200 bpm is a normal range for a human quietly sitting
     two_segment=True, # perform cyclical deformation deviding each respiratory cycle in 2 segments
     points_per_cycle=points_per_cycle, # set number of points per cycle
)

print('RespHRV features :')
print(resphrv_cycles)
print()
print('cyclic_cardiac_rate shape :')
print(cyclic_cardiac_rate.shape)

##############################################################################
# 
# RespHRV Features / Metrics
# ---------------------------------
# 
# `resphrv_cycles` is a dataframe containing one row per respiratory cycle and one
# column per heart rate related feature.
# 
# Some features are related to the position (index) or time of specific points
# within the cycle (see figure below for a graphical view of these timepoints/metrics):
# 
#    * `peak_index`: Index of the maximum heart rate during the current respiratory cycle `n` 
#      (usually during inspiration). The index refers to the sequential position of the sample 
#      in the entire instantaneous heart rate time series.
#    * `trough_index`: Index of the minimum heart rate during the current respiratory cycle `n` 
#      (usually during expiration).
#    * `peak_time`: Time in seconds of the maximum heart rate during the current respiratory cycle `n` 
#      (usually during inspiration).
#    * `trough_time`: Time in seconds of the minimum heart rate during the current respiratory cycle `n` 
#      (usually during expiration).
# 
# From these points, several derived features of interest (e.g., for statistical analysis) are computed. 
# Note that they are based on the expected heart rate dynamics under physiological conditions: 
# an increase in heart rate during inspiration (which already begins during the previous respiratory cycle) 
# and a decrease in heart rate during expiration (usually at the transition from inspiration to expiration).
# 
#    * `peak_value`: (units = those set in :py:func:`~physio.compute_resphrv`, default = `bpm`) 
#      Instantaneous heart rate at `peak_index` / `peak_time`, i.e., the maximum heart rate 
#      during the ongoing respiratory cycle (usually during inspiration). 
#    * `trough_value`: (units = those set in :py:func:`~physio.compute_resphrv`, default = `bpm`) 
#      Instantaneous heart rate at `trough_index` / `trough_time`, i.e., the minimum heart rate 
#      during the ongoing respiratory cycle (usually during expiration).
#    * `rising_amplitude`: (units = those set in :py:func:`~physio.compute_resphrv`, default = `bpm`) 
#      Difference in heart rate between the maximum of cycle `n` and the minimum of cycle `n-1` 
#      (see figure below).
#    * `decay_amplitude`: (units = those set in :py:func:`~physio.compute_resphrv`, default = `bpm`) 
#      Difference in heart rate between the maximum and the minimum of cycle `n` (see figure below).  
#      **This corresponds to "how much the heart rate decreases during the current respiratory cycle," 
#      usually at the transition from inspiration to expiration of cycle `n`. Under physiological 
#      conditions, this is considered the primary measure of RespHRV, but here it is computed for each cycle.**
#    * `rising_duration`: (units = `s`) Duration of the `rising_amplitude` period, i.e., 
#      (`peak_time` of cycle `n` – `trough_time` of cycle `n-1`). 
#    * `decay_duration`: (units = `s`) Duration of the `decay_amplitude` period, i.e., 
#      (`trough_time` of cycle `n` – `peak_time` of cycle `n`). 
#    * `rising_slope`: (units = `bpm/s` by default) Slope of the increase in heart rate, 
#      defined as `rising_amplitude` / `rising_duration`.
#    * `decay_slope`: (units = `bpm/s` by default) Slope of the decrease in heart rate, 
#      defined as `decay_amplitude` / `decay_duration`.
#
# .. image:: ./img/resphrv_features_doc_physio.png
#    :alt: RespHRV features
#    :align: center
#    :scale: 70%

print(f'{resphrv_cycles.shape[1]} computed heart-rate related features in {resphrv_cycles.shape[0]} respiratory cycles, these features being the following :')
pprint(resphrv_cycles.columns.to_list())

##############################################################################
# 
# Plot RespHRV Amplitude Over Time
# ----------------------
# 
# Using the `decay_amplitude` feature as a marker of cycle-by-cycle RespHRV amplitude, 
# we can plot the dynamics of RespHRV across respiratory cycles or over time 
# (see plot below).
# 
# Note that this participant shows an unusually large RespHRV amplitude (~ bpm). 
# For reference, typical values (in `bpm`) were computed from a sample of 
# N = 3798 cycles pooled from 29 human participants quietly sitting on a chair 
# for 10 minutes:  
# 
#    {'mean': 7.47, 'std': 7.14, '25%': 2.83, '50%': 5.54, '75%': 10.41, 'IQR': 7.58}



fig, axs = plt.subplots(nrows = 3, figsize = (8, 6), constrained_layout = True)
fig.suptitle('RespHRV according to the cycle number / time')

ax = axs[0]
ax.plot(range(resphrv_cycles.shape[0]), resphrv_cycles['decay_amplitude'], color = 'r', lw = 2) # plot decay_amplitude according to cycle number
ax.set_xlabel('Resp cycle number')
ax.set_ylabel('RespHRV amplitude (bpm)')
ax = axs[1]
ax.plot(resphrv_cycles['peak_time'], resphrv_cycles['decay_amplitude'], color = 'r', lw = 2) # plot decay_amplitude according to peak time
ax.set_xlabel('Time (s)')
ax.set_ylabel('RespHRV amplitude (bpm)')

ax = axs[2]
ax.plot(resphrv_cycles['peak_time'], resphrv_cycles['trough_value'], color = 'orangered', label = 'peak_value', lw = 3) # plot trough_value according to peak time
ax.plot(resphrv_cycles['peak_time'], resphrv_cycles['peak_value'], color = 'darkred', label = 'trough_value', lw = 3) # plot peak_value according to peak time
ax.fill_between(resphrv_cycles['peak_time'], resphrv_cycles['trough_value'], resphrv_cycles['peak_value'], color = 'r', alpha = 0.4, label = 'decay_amplitude') # fill between trough_value and peak_value  = RespHRV amplitude
ax.set_xlabel('Time (s)')
ax.set_ylabel('Heart Rate (bpm)')
ax.legend(loc = 'upper right', framealpha = 1)



##############################################################################
# 
# Plot Heart Rate Dynamics According to Respiratory Phase
# ----------------------
# 
# The function :py:func:`~physio.compute_resphrv` internally uses 
# :py:func:`~physio.deform_traces_to_cycle_template` to deform the instantaneous 
# heart rate vector according to inspiratory/expiratory time points 
# (see :ref:`sphx_glr_examples_example_04_cyclic_deformation.py` for more information 
# about cyclical deformation).
# 
# As a result, :py:func:`~physio.compute_resphrv` returns `cyclic_cardiac_rate`, 
# a 2D NumPy array of shape (`n_resp_cycles`, `points_per_cycle`). Each cell of this 
# array contains a heart rate value, expressed in the units set in 
# :py:func:`~physio.compute_resphrv` (default = `bpm`).  
# 
# This matrix provides a detailed view of heart rate dynamics at each phase point 
# of every respiratory cycle. Specifically, `points_per_cycle` defines the number 
# of respiratory phase points to which the heart rate is reinterpolated.  
# 
# Using this representation, one can plot the cycle-by-cycle dynamics of heart rate 
# during each respiratory cycle (shown as overlapping black traces in the figure below) 
# along with the average heart rate dynamics (shown as the thick orange trace), 
# relative to the normalized respiratory phase (normalized from 0 to 1, but it can 
# also be expressed from 0 to 2π or –π to π).



cycle_ratio = resp_cycles['cycle_ratio'].mean() # get the average cycle ratio, internally used by physio.compute_resphrv as the segment_ratios, if two_segment = True

one_cycle = np.arange(points_per_cycle) / points_per_cycle # compute normalized respiratory phase vector
fig, ax = plt.subplots()
ax.plot(one_cycle, cyclic_cardiac_rate.T, color='k', alpha=.3) # plot heart rate dynamic for each cycle, overlapped, in black, according to resp phase
ax.plot(one_cycle, np.mean(cyclic_cardiac_rate, axis=0), color='darkorange', lw=3) # # plot average heart rate dynamic, in orange, according to resp phase. The mean is computed along axis 0 meaning along the resp cycles axis.
ax.axvspan(0, cycle_ratio, color='g', alpha=0.3) # plot a vertical green span from 0 to the mean cycle_ratio (meaning the mean normalized transition from inspi to expi) to display inspiration period
ax.axvspan(cycle_ratio, 1, color='r', alpha=0.3) # plot a vertical red span from the mean cycle_ratio to 1 to display expiration period
ax.set_xlabel('Respi phase (0-1)')
ax.set_ylabel('Heart rate (bpm)')
ax.set_xlim(0, 1)
ax.text(0.2, 60, 'inhalation', ha='center', color='g')
ax.text(0.85, 60, 'exhalation', ha='center', color='r')
ax.set_title('Cycle-by-cycle plot of Heart Rate Dynamic according to respiratory phase\nRespHRV = vertical range of these epochs')


plt.show()

##############################################################################
# 
# Plot heart rate dynamic according to time and respiratory phase
# ----------------------
# 
# A nice way of seeing RespHRV can be a 2D plot showing evolution along time / resp cycles of the heart rate dynamics along each respiratory cycle

fig, axs = plt.subplots(nrows = 2, figsize = (8, 6), constrained_layout = True, sharex = False)
fig.suptitle('Cycle-by-cycle 2D plot of Heart Rate Dynamic according to time * respiratory phase\nRespHRV = color range of color intensity', fontsize=  13)

ax = axs[0]
im = ax.pcolormesh(range(resp_cycles.shape[0]), one_cycle, cyclic_cardiac_rate.T, cmap = 'viridis') # plot 2D view of heart rate dynamic for each cycle (in abscissa) and each phase point (in ordinate)
fig.colorbar(im, ax=ax).set_label('Raw Heart Rate (bpm)')
ax.set_title('Raw cyclic_cardiac_rate')
ax.set_xlabel('Resp cycle Number')

ax = axs[1]
cyclic_cardiac_rate_centered = cyclic_cardiac_rate - np.mean(cyclic_cardiac_rate, axis = 1)[:,None] # Center each heart rate epoch by subtracting to them the mean of each one
# This can be useful to focus just on heart rate by-cycle variations and remove slow heart rate drifts than can bother the viewing of RespHRV

im = ax.pcolormesh(resp_cycles['inspi_time'], one_cycle, cyclic_cardiac_rate_centered.T, cmap = 'viridis') # plot 2D view of heart rate dynamic for each cycle (in abscissa) and each phase point (in ordinate)
fig.colorbar(im, ax=ax).set_label('Heart Rate centered (bpm)')
ax.set_title('cyclic_cardiac_rate has been centered\nby the mean computed along axis = 1, meaning phase axis')
ax.set_xlabel('Time (seconds) (inspi_time of the resp_cycles)\n(~same than upper plot but in time instead of N° of cycle)')

for ax in axs:
    ax.set_ylabel('Respi phase (0-1)')
    ax.axhline(cycle_ratio, color = 'r') # plot a horizontal red line at the mean cycle_ratio
    ax.text(1, cycle_ratio + 0.05, 'Inspi-Expi transition', ha='left', color='r')

plt.show()


##############################################################################
# 
# Explore Covariations Between RespHRV and Respiratory Features
# ----------------------
# 
# `resp_cycles` and `resphrv_cycles` contain the same number of rows, 
# since they describe respiratory and heart rate features for each respiratory cycle 
# (`nrows = n_cycles`).  
# 
# This makes it possible to analyze covariations between cycle-by-cycle 
# respiratory features and RespHRV features.  
# 
# For such an analysis, ensure that your dataset includes a sufficient number 
# of respiratory cycles and/or enough variation in respiratory features or regimes. 
# Otherwise, there may not be enough variability to meaningfully explore 
# the covariations.  
# 
# In this example, only 5 minutes of respiration and ECG signals were recorded 
# in a quiet resting state. Therefore, variability and the number of cycles 
# are limited. Nevertheless, we perform such an analysis here for demonstration purposes.


alpha = 0.05 # regression line will be plotted if the regression fit p-value is lower than this alpha threshold

# define a function to filter outliers
def filter_outliers(x, y): 
   _factor = 3

   med, mad = physio.compute_median_mad(x[~np.isnan(x)])
   x_keep = (x >= med - _factor * mad) & (x <= med + _factor * mad)

   med, mad = physio.compute_median_mad(y[~np.isnan(y)])
   y_keep = (y >= med - _factor * mad) & (y <= med + _factor * mad)

   and_keep = x_keep & y_keep
   return x[and_keep], y[and_keep]

resp_sel_metrics = ['inspi_amplitude','expi_amplitude','inspi_duration','expi_duration','cycle_duration'] # select some respiratory features to study
resphrv_sel_metrics = ['decay_amplitude','rising_amplitude','decay_duration','rising_duration'] # select some RespHRV features to study

nrows = len(resphrv_sel_metrics)
ncols = len(resp_sel_metrics)

fig, axs = plt.subplots(nrows, ncols, figsize = (ncols * 3, nrows * 3), constrained_layout = True)
fig.suptitle(f'Covariations between RespHRV and Respiratory Features\nSuch analysis need large amount of resp cycles, even more than the {resp_cycles.shape[0]} available cycles of this example.', fontsize = 15)

for c, resp_metric in enumerate(resp_sel_metrics):
  for r, resphrv_metric in enumerate(resphrv_sel_metrics):
    ax = axs[r,c]

    x, y = filter_outliers(resp_cycles[resp_metric].values, resphrv_cycles[resphrv_metric].values) # filter paired outliers

    res = stats.linregress(x,y) # linear regression from scipy.stats

    ax.set_title(f'r : {round(res.rvalue,2)} -  r2 = {round(res.rvalue**2,2)} - p = {round(res.pvalue, 2)}')
    if res.pvalue < alpha: # plot regression line if p-value < alpha
      ax.plot(x, res.intercept + res.slope*x, color = 'r')
    ax.scatter(x, y, color = 'k')
    ax.set_ylabel(resphrv_metric)
    ax.set_xlabel(resp_metric)

plt.show()
