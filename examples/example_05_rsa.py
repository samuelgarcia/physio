'''
RSA tutorial
============

Respiratory sinus arrhythmia (RSA) can be analysed with the physio toolbox with
an innovative method to extract parameters of the heart rate dynamic on a respiratory
cycle-to-cycle basis.

The method consists of:
  * detect respiratory cycle
  * detect ECG peaks
  * compute instanteneous heart rate in BPM
  * extract parameters in this heart timeseries for each respiratory cycle
  * use cyclic deformation of this heart rate signal and stack all cycles.

This method has 2 important advantages:
  * the dynamic of th RSA can be finely analysed
  * features of RSA can be analysed using respiratory cycle-by-cycle

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

raw_resp = np.load('resp1.npy')
raw_ecg = np.load('ecg1.npy')
srate = 1000.
times = np.arange(raw_resp.size) / srate


##############################################################################
# 
# Get respiratory cycle and ECG peaks
# -----------------------------------
#  
#

resp, resp_cycles = physio.compute_respiration(raw_resp, srate, parameter_preset='human_airflow')
ecg, ecg_peaks = physio.compute_ecg(raw_ecg, srate, parameter_preset='human_ecg')


##############################################################################
# 
# Compute RSA
# -----------
#  
# This is done with one unique function that returns:
#   
#   * One dataframe with all RSA features
#   * The cyclic deformed cardiac rate

points_per_cycle = 50

rsa_cycles, cyclic_cardiac_rate = physio.compute_rsa(
    resp_cycles,
     ecg_peaks,
     srate=10.,
     two_segment=True,
     points_per_cycle=points_per_cycle,
)

some_features = ['rising_amplitude', 'decay_amplitude', 'rising_duration', 'decay_duration', 'rising_slope', 'decay_slope']
print(rsa_cycles[some_features].head(9))


##############################################################################
# 
# Plot RSA cycle dynamic
# ----------------------
# 
# Here we also plot the average ratio of inspiration duration to cycle duration

# this is the average ratio
inspi_expi_ratio = np.mean(resp_cycles['inspi_duration'] / resp_cycles['cycle_duration'])

one_cycle = np.arange(points_per_cycle) / points_per_cycle
fig, ax = plt.subplots()
ax.plot(one_cycle, cyclic_cardiac_rate.T, color='k', alpha=.3)
ax.plot(one_cycle, np.mean(cyclic_cardiac_rate, axis=0), color='darkorange', lw=3)
ax.axvspan(0, inspi_expi_ratio, color='g', alpha=0.3)
ax.axvspan(inspi_expi_ratio, 1, color='r', alpha=0.3)
ax.set_xlabel('One respiratory cycle')
ax.set_ylabel('Heart rate [bpm]')
ax.set_xlim(0, 1)
ax.text(0.2, 60, 'inhalation', ha='center', color='g')
ax.text(0.85, 60, 'exhalation', ha='center', color='r')
ax.set_title('All RSA cycle streched to resp cycles')


plt.show()
