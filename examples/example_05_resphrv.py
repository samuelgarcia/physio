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
from pathlib import Path
from pprint import pprint

import physio



##############################################################################
# 
# Read data
# ----------
#  
# For this tutorial, we will use an internal file stored in NumPy format for demonstration purposes.
# See :ref:`sphx_glr_examples_example_01_getting_started.py`, first section, for a description of 
# the capabilities of :py:mod:`physio` for reading raw data formats.


raw_resp = np.load('resp_airflow1.npy') # load respi
raw_ecg = np.load('ecg1.npy') # load ecg
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
#    * `resp_cycles`: pd.DataFrame, output of the function :py:func:`~physio.compute_respiration`
#    * `ecg_peaks`: pd.DataFrame, output of the function :py:func:`~physio.compute_ecg`
#    * `srate`: int or float. (optional) Sampling rate used for interpolation to get an instantaneous heart rate vector from RR intervals. 100 Hz is safe for both animal and human. For human 10 also works.
#    * units : str (bpm / s / ms / Hz), sets the output units (optional, default = 'bpm').
#    * limits : list or None, (optional) range in the chosen units for removing outliers (e.g., [30, 200] to exclude bpm values outside this range). Default is None, meaning no cleaning..
#    * two_segment : bool, (optional, default = True), True or False, to perform cyclical deformation deviding each respiratory cycle in 2 (if True) segments or 1 (if False).  See :ref:`sphx_glr_examples_example_04_cyclic_deformation.py` for more informations.
#    * points_per_cycle : int, (optional, default = 50), number of points per cycle used for linear resampling during cyclical deformation (see :ref:`sphx_glr_examples_example_04_cyclic_deformation.py`).
#
# When called, :py:func:`~physio.compute_resphrv` performs the following:
#    * Computes instantaneous heart rate (IHR) vector from RR intervals computed from `ecg_peaks`.
#    * Perform cyclic deformation of the IHR vector according to respiratory time points (returns a NumPy array: `cyclic_cardiac_rate` of shape (n_resp_cycles * points_per_cycle))
#    * Computes Heart-Rate features for each respiratory cycle period (returns a pd.DataFrame array: `resphrv_cycles`)

points_per_cycle = 50

resphrv_cycles, cyclic_cardiac_rate = physio.compute_resphrv(
    resp_cycles,
     ecg_peaks,
     srate=10.,
     two_segment=True,
     points_per_cycle=points_per_cycle,
)

some_features = ['rising_amplitude', 'decay_amplitude', 'rising_duration', 'decay_duration', 'rising_slope', 'decay_slope']
print(resphrv_cycles[some_features].head(9))


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
