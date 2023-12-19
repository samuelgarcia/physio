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
# Detect ECG R peaks: quick way
# -----------------------------
#
# The fastest way is to use compute_ecg() using a predefined parameter
# preset. Here is a simple example.


raw_ecg = np.load('ecg1.npy')
srate = 1000.
times = np.arange(raw_ecg.size) / srate

ecg, ecg_peaks = physio.compute_ecg(raw_ecg, srate, parameter_preset='human_ecg')

r_peak_ind = ecg_peaks['peak_index'].values

fig, axs = plt.subplots(nrows=2, sharex=True)
ax = axs[0]
ax.plot(times, raw_ecg)
ax.scatter(times[r_peak_ind], raw_ecg[r_peak_ind], marker='o', color='magenta')
ax.set_ylabel('raw ecg')

ax = axs[1]
ax.plot(times, ecg)
ax.scatter(times[r_peak_ind], ecg[r_peak_ind], marker='o', color='magenta')
ax.set_ylabel('ecg')

ax.set_xlim(95, 125)

##############################################################################
# 
# Detect ECG R peaks: Parameters tuning
# -------------------------------------
# 
# Here is a simple recipe to change some predefined parameters.
# We change here some filtering parameters.

# get paramseters predefined set for 'human_ecg'
# this is a nested dict of parameter of every step
parameters = physio.get_ecg_parameters('human_ecg')
pprint(parameters)
# lets change on parameter in the structure
parameters['preprocess']['band'] = [2., 40.]
parameters['preprocess']['ftype'] = 'bessel'
parameters['preprocess']['order'] = 4
pprint(parameters)

ecg, ecg_peaks = physio.compute_ecg(raw_ecg, srate, parameters=parameters)

r_peak_ind = ecg_peaks['peak_index'].values

fig, ax = plt.subplots()
ax.plot(times, ecg)
ax.scatter(times[r_peak_ind], ecg[r_peak_ind], marker='o', color='magenta')
ax.set_ylabel('ecg')
ax.set_xlim(95, 125)



##############################################################################
# 
# ECG: compute metrics
# --------------------
#


metrics = physio.compute_ecg_metrics(ecg_peaks, min_interval_ms=500., max_interval_ms=2000.)
print(metrics)

##############################################################################
# 
# ECG : compute instantaneous rate
# --------------------------------
#
# The RR-interval (aka rri) time series is a common tool to analyse the heart rate variability (hrv).
# This is equivalent to computing the instantaneous heart rate.
# Heart rate [bpm] = 1 / rri * 60
#
# Most people use rri in ms, we feel that use heart rate in bpm is more intuitive. 
# With bpm an increase in the curve = heart acceleration. 
# With ms an increase in the curve = heart decceleration. 
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
# ECG: compute hrv spectrum
# -------------------------
#
# 

freqency_bands = {'lf': (0.04, .15), 'hf' : (0.15, .4)}
psd_freqs, psd, psd_metrics = physio.compute_hrv_psd(
    ecg_peaks,
    sample_rate=100.,
    limits=None,
    units='bpm',
    freqency_bands=freqency_bands,
    window_s=250.,
    interpolation_kind='cubic',
)

print(psd_metrics)
fig, ax = plt.subplots()
# ax.semilogy(psd_freqs, psd)
ax.plot(psd_freqs, psd)
colors = {'lf': '#B8860B', 'hf' : '#D2691E'}
for name, freq_band in freqency_bands.items():
    ax.axvspan(*freq_band, alpha=0.1, color=colors[name], label=f'{name} : {psd_metrics[name]}')
ax.set_xlim(0, 0.6)
ax.set_xlabel('freq [Hz]')
ax.set_ylabel('HRV PSD')
ax.legend()

plt.show()
