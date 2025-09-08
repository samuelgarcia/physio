'''
ECG tutorial
============

'''

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pprint

import physio


##############################################################################
# 
# Detect ECG R peaks. The faster way: using `parameter_preset`.
# ------------------------------------------
#
# The fastest way to process ECG with :py:mod:`physio` is to use :py:func:`~physio.compute_ecg` which is a high-level wrapper function that simplifies ECG signal analysis.
# To use this function, you must provide:
#    * `raw_ecg` : the raw ECG signal as a NumPy array
#    * `srate` : the sampling rate of the ECG signal
#    * `parameter_preset` : a string specifying the type of ECG data, which determines the set of parameters used for processing. Can be one of: `human_ecg` or `rat_ecg`.
#
# When called, :py:func:`~physio.compute_ecg` performs the following:
#    * Preprocesses the ECG signal (returns a NumPy array: `ecg`). By default, `ecg` is normalized.
#    * Detects R peaks (returns a pd.DataFrame: `ecg_peaks`)
# 
# **Warning:** The orientation of the `raw_ecg` trace is important (multiply it by -1 for reversing it if necessary). 
# R peaks must point upward for the highest probability of detection by :py:func:`~physio.compute_ecg`. 
# Sometimes the R and S peaks of the QRS complex are equivalent, so orientation does not matter, 
# because either the R or S peak will be detected and used to mark heartbeats.
# 
# For this tutorial, we will use an internal file already stored in NumPy format for demonstration purposes.

raw_ecg = np.load('ecg1.npy') # load ecg
srate = 1000. # our example signals have been recorded at 1000 Hz

times = np.arange(raw_ecg.size) / srate # build time vector

ecg, ecg_peaks = physio.compute_ecg(raw_ecg, srate, parameter_preset='human_ecg') # set 'human_ecg' as preset because example ecg is from human

# ecg_peaks is a dataframe containing 2 columns : 1) the indices of detection and 2) the times of detections of the R peaks
print(ecg_peaks)

r_peak_ind = ecg_peaks['peak_index'].values # get index of detected R peaks

linewidth = 0.8
fig, axs = plt.subplots(nrows=2, sharex=True)
ax = axs[0]
ax.plot(times, raw_ecg, lw = linewidth)
ax.scatter(times[r_peak_ind], raw_ecg[r_peak_ind], color="#C3CB22", label = 'R peak')
ax.set_ylim(0.0145, 0.0175)
ax.set_ylabel('Amplitude (V)')
ax.set_title('R peak detection, plotted on raw ECG')
ax.legend(loc = 'upper right')

ax = axs[1]
ax.plot(times, ecg, lw = linewidth)
ax.scatter(times[r_peak_ind], ecg[r_peak_ind], color='#A8AE27', label = 'R peak')
ax.set_ylabel('Amplitude (AU)')
ax.set_title('R peak detection, plotted on preprocessed ECG')
ax.legend(loc = 'upper right')

ax.set_xlabel('Time (s)')
ax.set_xlim(95, 125)

##############################################################################
# 
# What is `parameter_preset` ?
# ---------------------------
# 
# Using `parameter_preset` tells :py:func:`~physio.compute_ecg` to process ECG
# according to a predefined set of parameters already optimized by :py:mod:`physio`.
# 
# To get an idea of the default parameters used in `parameter_preset`, 
# you can call :py:func:`~physio.get_ecg_parameters`.
# 
# Here is an example showing the set of parameters applied in the case of a human ECG signal.

parameters = physio.get_ecg_parameters('human_ecg') # parameters is a nested dictionary of parameters used at each processing step.
pprint(parameters) # pprint to "pretty print"

##############################################################################
# 
# Tuning parameters if unsatisfied
# ---------------------------------
# 
# Variability during data acquisition (subject, acquisition system) can affect the recorded ECG signal.
# Such variability may make some predefined parameters of :py:mod:`physio` inappropriate. 
# 
# In this situation, you can tune certain parameters by re-assigning values to the keys of the `parameters` dictionary.
# You may also tune multiple parameters at once if necessary. 
# **To fine-tune parameters properly, a good understanding of each parameter's role is required. 
# For this reason, we have dedicated a whole section to this topic — see the "Parameters" section.**
# 
# For example, here we modify 3 parameters: 
#    * the amplitude above which R peak can be detected
#    * the minimum possible duration in milliseconds between two consecutive R peaks
#    * the frequency band of the filter

# let's change 3 parameters in the nested structure ...
parameters['peak_detection']['thresh'] = 4 # at the key "peak_detection" and the sub-key "thresh", the default values is 'auto', meaning an automatically computed threshold of detection. Here we replace it to 4 to allow for peaks at least higher than 4 to be detected.
parameters['peak_clean']['min_interval_ms'] = 300 # at the key "peak_clean" and the sub-key "min_interval_ms", the default values is 400 milliseconds. Here we replace it to 300 to allow smaller RR intervals to be detected, meaning faster heart rate.
parameters['preprocess']['band'] = [2., 25.] # at the key "preprocess" and the sub-key "band", the default values is [5., 45.] Hz. Here we replace it to [2., 40.] Hz to target a slightly lower frequency band.
pprint(parameters)

ecg, ecg_peaks = physio.compute_ecg(raw_ecg, srate, parameters=parameters) # preset_parameters = None in this case, because parameters is now explicitly defined

print(ecg_peaks)

r_peak_ind = ecg_peaks['peak_index'].values # get index of detected R peaks

fig, ax = plt.subplots()
ax.plot(times, ecg)
ax.scatter(times[r_peak_ind], ecg[r_peak_ind], marker='o', color='magenta')
ax.set_ylabel('Amplitude (AU)')
ax.set_title('R peak detection, plotted on preprocessed ECG')
ax.set_xlabel('Time (s)')
ax.set_xlim(95, 125)



##############################################################################
# 
# ECG: compute time-domain heart rate variability (HRV) metrics
# --------------------
#


metrics = physio.compute_ecg_metrics(ecg_peaks, min_interval_ms=500., max_interval_ms=2000.)
print(metrics)

##############################################################################
# 
# ECG : compute instantaneous rate
# --------------------------------
#
# The RR-interval (aka rri) time series is a common tool to analyse the heart rate variability (HRV).
# This is equivalent to compute the instantaneous heart rate.
# Heart rate [bpm] = 1 / rri * 60
#
# Most people use rri in ms, we feel that the use of heart rate in bpm is more intuitive. 
# With bpm unit, an increase in the curve means heart rate acceleration. 
# With ms unit, an increase in the curve means heart rate deceleration. 
#
# Feel free to use the units you prefer (bpm or ms)

new_times = times[::10]
instantaneous_rate = physio.compute_instantaneous_rate(
    ecg_peaks,
    new_times,
    limits=None,
    units='bpm',
    interpolation_kind='linear',
)
rri = physio.compute_instantaneous_rate(
    ecg_peaks,
    new_times,
    limits=None,
    units='ms',
    interpolation_kind='linear',
)

fig, axs = plt.subplots(nrows=2, sharex=True)
ax = axs[0]
ax.plot(new_times, instantaneous_rate)
ax.set_ylabel('heart rate [bpm]')
ax = axs[1]
ax.plot(new_times, rri)
ax.set_ylabel('rri [ms]')
ax.set_xlabel('time [s]')
ax.set_xlim(100, 150)

##############################################################################
# 
# ECG: compute frequency-domain heart rate variability (HRV) metrics
# -------------------------
#
# 

frequency_bands = {'lf': (0.04, .15), 'hf' : (0.15, .4)}
psd_freqs, psd, psd_metrics = physio.compute_hrv_psd(
    ecg_peaks,
    sample_rate=100.,
    limits=None,
    units='bpm',
    frequency_bands=frequency_bands,
    window_s=250.,
    interpolation_kind='cubic',
)

print(psd_metrics)
fig, ax = plt.subplots()
# ax.semilogy(psd_freqs, psd)
ax.plot(psd_freqs, psd)
colors = {'lf': '#B8860B', 'hf' : '#D2691E'}
for name, freq_band in frequency_bands.items():
    ax.axvspan(*freq_band, alpha=0.1, color=colors[name], label=f'{name} : {psd_metrics[name]}')
ax.set_xlim(0, 0.6)
ax.set_xlabel('freq [Hz]')
ax.set_ylabel('HRV PSD')
ax.legend()

plt.show()
