'''
ECG example
===========

'''


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pprint

import physio

##############################################################################
# 
# physio.ecg
# 
# ----------
#
# 
#  

##############################################################################
# 
# ECG : quick way
# 
# ---------------
#
#  The fastest way is to use compute_ecg() using predefine parameters set
#  here a simple example


raw_ecg = np.load('ecg1.npy')
srate = 1000.
times = np.arange(raw_ecg.size) / srate


ecg, ecg_peaks = physio.compute_ecg(raw_ecg, srate, parameter_set='simple_ecg')

fig, axs = plt.subplots(nrows=2, sharex=True)
ax = axs[0]
ax.plot(times, raw_ecg)
ax.scatter(times[ecg_peaks], raw_ecg[ecg_peaks], marker='o', color='magenta')
ax.set_ylabel('raw ecg')

ax = axs[1]
ax.plot(times, ecg)
ax.scatter(times[ecg_peaks], ecg[ecg_peaks], marker='o', color='magenta')
ax.set_ylabel('ecg')

ax.set_xlim(95, 125)

##############################################################################
# 
# Parameters tuning
# 
# -----------------------
# 
# Here a simple recipe to change some predefined parameters
# We change here some filtering parameters

# get paramseters predefined set for 'simple_ecg'
# this is a nested dict of parameter of every step
parameters = physio.get_ecg_parameters('simple_ecg')
pprint(parameters)
# lets change on parameter in the structure
parameters['preprocess']['band'] = [2., 40.]
parameters['preprocess']['ftype'] = 'bessel'
parameters['preprocess']['order'] = 4
pprint(parameters)

ecg, ecg_peaks = physio.compute_ecg(raw_ecg, srate, parameters=parameters)

fig, ax = plt.subplots()
ax.plot(times, ecg)
ax.scatter(times[ecg_peaks], ecg[ecg_peaks], marker='o', color='magenta')
ax.set_ylabel('ecg')
ax.set_xlim(95, 125)



##############################################################################
# 
# ECG : compute metrics
# 
# ---------------------
#


metrics = physio.compute_ecg_metrics(ecg_peaks, srate, min_interval_ms=500., max_interval_ms=2000.)
print(metrics)

##############################################################################
# 
# ECG : compute instantaneous rate aka hrv
# 
# ----------------------------------------
#



rate_times = times[::10]
peak_times = ecg_peaks / srate
instantaneous_rate = physio.compute_instantaneous_rate(peak_times, rate_times, limits=None, units='bpm', interpolation_kind='linear')

fig, ax = plt.subplots()
ax.plot(rate_times, instantaneous_rate)
ax.set_xlabel('time [s]')
ax.set_ylabel('hrv [bpm]')

##############################################################################
# 
# ECG : compute hrv spectrum
# --------------------------
#
# 

ecg_duration_s = times[-1]
freqency_bands = {'lf': (0.04, .15), 'hf' : (0.15, .4)}
psd_freqs, psd, psd_metrics = physio.compute_hrv_psd(peak_times, ecg_duration_s,  sample_rate=100., limits=None, units='bpm',
                                        freqency_bands=freqency_bands,
                                        window_s=250., interpolation_kind='cubic')

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