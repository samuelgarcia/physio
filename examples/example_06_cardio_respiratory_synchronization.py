'''

Cardio respiratory synchronization
==================================


'''


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pprint

import physio



##############################################################################
# 
# Read data
# ---------
#  
# 

raw_resp = np.load('resp2.npy')
raw_ecg = np.load('ecg2.npy')
srate = 1000.
times = np.arange(raw_resp.size) / srate


##############################################################################
# 
# Compute respiration cycles and ecg R peaks
# ------------------------------------------
#  
# 


ecg, ecg_peaks = physio.compute_ecg(raw_ecg, srate)
resp, resp_cycles = physio.compute_respiration(raw_resp, srate)


##############################################################################
# 
# First plot
# ------------------------------------------
#  
# 



fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(15, 10))
ax = axs[0]
ax.plot(times, resp)
ax.scatter(resp_cycles['inspi_time'], resp[resp_cycles['inspi_index']], color='g')
ax.scatter(resp_cycles['expi_time'], resp[resp_cycles['expi_index']], color='r')
for t in ecg_peaks['peak_time']:
    ax.axvline(t, color='m')

ax = axs[1]
ax.plot(times, ecg)
ax.scatter(ecg_peaks['peak_time'], ecg[ecg_peaks['peak_index']], color='m')

ax.set_xlim(0,  40)


##############################################################################
# 
# Ecg peaks transform in resp cycle phase
# ---------------------------------------
#  
# sphinx_gallery_thumbnail_number = 2


inspi_ratio = resp_cycles['cycle_ratio'].median()

cycle_times = resp_cycles[['inspi_time', 'expi_time', 'next_inspi_time']].values

rpeak_phase = physio.time_to_cycle(ecg_peaks['peak_time'].values, cycle_times, segment_ratios=[inspi_ratio])


count, bins = np.histogram(rpeak_phase % 1, bins=np.linspace(0, 1, 101))

fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(8, 6))
ax = axs[0]
ax.scatter(rpeak_phase % 1, np.floor(rpeak_phase))
ax.set_ylabel('resp cycle')

ax = axs[1]
ax.bar(bins[:-1], count, width=bins[1] - bins[0], align='edge')
ax.set_xlabel('resp phase')
ax.set_ylabel('count R peaks')

for ax in axs:
    ax.axvline(0, color='g', label='inspiration', alpha=.6)
    ax.axvline(inspi_ratio, color='r', label='expiration', alpha=.6)
    ax.axvline(1, color='g', label='next_inspiration', alpha=.6)
ax.legend()
ax.set_xlim(-0.01, 1.01)

##############################################################################
# 
# Cross-correlogram between expiration times and ECG peak times
# -------------------------------------------------------------
# 

bins = np.linspace(-3, 3, 100)


count, bins = physio.crosscorrelogram(ecg_peaks['peak_time'].values, 
                               resp_cycles['expi_time'].values,
                               bins=bins)

fig, axs = plt.subplots(nrows=2, sharex=True)
ax = axs[0]
ax.bar(bins[:-1], count, align='edge', width=bins[1] - bins[0])
ax.set_xlabel('time lag')
ax.set_ylabel('count')
ax.axvline(0, color='r', label='expiration', alpha=.6)
ax.legend()


ax = axs[1]
count, bins = physio.crosscorrelogram(ecg_peaks['peak_time'].values, 
                               resp_cycles['inspi_time'].values,
                               bins=bins)
ax.bar(bins[:-1], count, align='edge', width=bins[1] - bins[0])
ax.set_xlabel('time lag')
ax.set_ylabel('count')
ax.axvline(0, color='g', label='inspiration', alpha=.6)
ax.legend()


plt.show()

ax.set_xlim(-0.01, 1.01)