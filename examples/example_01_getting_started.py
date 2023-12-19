'''
Getting started tutorial
========================

Here is a quick overview of :py:mod:`physio`
'''

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import physio

##############################################################################
# 
# physio module overview
# ----------------------
# 
# :py:mod:`physio` has utility functions for reading some formats like:
#  
#    * micromed
#    * brainvision
#
# Here, we use an internal file in the numpy format for the demo

raw_resp = np.load('resp1.npy')
raw_ecg = np.load('ecg1.npy')
srate = 1000.
times = np.arange(raw_resp.size) / srate

fig, axs = plt.subplots(nrows=2, sharex=True)
ax = axs[0]
ax.plot(times, raw_resp)
ax.set_ylabel('raw resp')

ax = axs[1]
ax.plot(times, raw_ecg)
ax.set_ylabel('raw ECG')

ax.set_xlim(185, 225)


##############################################################################
# 
# Analyse respiration
# -------------------
# 
# :py:func:`~physio.compute_respiration` is an easy function to:
#
#    * preprocess the respiration signal
#    * compute cycle
#    * compute cycle features


resp, resp_cycles = physio.compute_respiration(raw_resp, srate)

fig, ax = plt.subplots()
ax.plot(times, raw_resp)
ax.plot(times, resp)
ax.set_xlim(185, 225)

##############################################################################
# 
# repiration cycles and features
# ------------------------------
#  
# resp_cycles is a dataframe containing all respiration cycles as rows and columns.
# It contains features like duration, amplitudes, durations
# 

print(resp_cycles.shape)
print(resp_cycles.columns)

columns = ['cycle_duration', 'inspi_volume', 'expi_volume', 'total_amplitude' ]
resp_cycles[columns].plot(kind='hist', subplots=True, sharex=False, layout=(2, 2), bins=50)

resp_cycles


##############################################################################
# 

inspi_ind = resp_cycles['inspi_index'].values
expi_ind = resp_cycles['expi_index'].values

fig, ax = plt.subplots()
ax.plot(times, resp)
ax.scatter(times[inspi_ind], resp[inspi_ind], color='green')
ax.scatter(times[expi_ind], resp[expi_ind], color='red')
ax.set_xlim(185, 225)



##############################################################################
# 
# Analyse ECG
# -----------
#  
# :py:func:`~physio.compute_ecg` is an easy function to:
#
#     * Preprocess the ECG signal output, which is normalized by default
#     * Detect R peaks


ecg, ecg_peaks = physio.compute_ecg(raw_ecg, srate)

r_peak_ind = ecg_peaks['peak_index'].values

fig, axs = plt.subplots(nrows=2, sharex=True)
ax = axs[0]
ax.plot(times, raw_ecg)
ax.scatter(times[r_peak_ind], raw_ecg[r_peak_ind], color='#A8AE27')

ax = axs[1]
ax.plot(times, ecg)
ax.scatter(times[r_peak_ind], ecg[r_peak_ind], color='#A8AE27')
ax.set_xlim(185, 225)


##############################################################################
# 
# ECG metrics
# -----------
# 
# :py:func:`~physio.compute_ecg_metrics` is a simple function to compute temporal based metrics around ECG
# 
#
# We can visualize theses metrics and the RR interval distribution.


ecg_metrics = physio.compute_ecg_metrics(ecg_peaks)

fig, ax = plt.subplots()
ax.hist(np.diff(ecg_peaks['peak_time']) * 1000., bins=np.arange(0, 1400, 10), alpha=0.5)
ax.axvline(ecg_metrics['HRV_Mean'], color='orange', label='HRV_Mean')
ax.axvline(ecg_metrics['HRV_Median'], color='violet', label='HRV_Median')
ax.set_xlabel('HRV [ms]')
ax.legend()
print(ecg_metrics)




##############################################################################
# 
# Cyclic deformation
# ------------------
# 
# :py:func:`~physio.deform_traces_to_cycle_template` is a tool to deform traces
# to a cycle template by stretching with linear resampling to a fixed number of
# points per cycle.
#
# This is helpfull to check if a signal is driven by a cyclic event like respiration.
# 
# Here, we deform the signal trace by "itself" : the respiration cycle.
# This leads to an average respiration template.
#
# Importantly, this can be done using one or several segment inside the cycle.

# here we have 3 time per cycle so 2 segments
cycle_times = resp_cycles[['inspi_time', 'expi_time', 'next_inspi_time']].values
deformed_resp_1seg = physio.deform_traces_to_cycle_template(resp, times, cycle_times,
                                                points_per_cycle=40, segment_ratios=0.4,
                                                output_mode='stacked')
print(deformed_resp_1seg.shape, cycle_times.shape)

# here we have 2 time per cycle so 1 segment
cycle_times = resp_cycles[['inspi_time', 'next_inspi_time']].values
deformed_resp_2seg = physio.deform_traces_to_cycle_template(resp, times, cycle_times,
                                                points_per_cycle=40, segment_ratios=None,
                                                output_mode='stacked')
print(deformed_resp_2seg.shape, cycle_times.shape)

fig, axs = plt.subplots(ncols=2, sharey=True)
physio.plot_cyclic_deformation(deformed_resp_1seg, segment_ratios=None, two_cycles=False, ax=axs[0])
axs[0].set_title('Deformation 2 segments')
physio.plot_cyclic_deformation(deformed_resp_2seg, segment_ratios=None, two_cycles=False, ax=axs[1])
axs[1].set_title('Deformation 1 segment')


##############################################################################
# 
# Cyclic deformation on ECG
# -------------------------
# 
# Lets use the same for ECG trace
# 
# We can also use a simple vector in this case it is converted a a 1 segment case.

# This is 1 segment
cycle_times = ecg_peaks['peak_time'].values
deformed_ecg = physio.deform_traces_to_cycle_template(ecg, times, cycle_times,
                                                points_per_cycle=300, segment_ratios=None,
                                                output_mode='stacked')
print(deformed_ecg.shape, cycle_times.shape)

fig, ax = plt.subplots()
physio.plot_cyclic_deformation(deformed_ecg, two_cycles=True, ax=ax)
ax.set_title('two ECG cycle avaerage')

##############################################################################
# 



plt.show()
