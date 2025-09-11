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
# Reading data
# ------------
# 
# :py:mod:`physio` aims to process physiological signals in NumPy format.
# Therefore, the first step is to load your data into a NumPy array.
# :py:mod:`physio` provides utility functions for reading certain formats, such as:
#
#    * micromed
#    * brainvision
#
# For example, cardiac activity recorded by a BrainVision system with the channel name "ECG" can be read using
# :py:func:`~physio.read_one_channel` by specifying: 
#
#    * the path to the file ("/path/to/the/file.vhdr")
#    * the format ("brainvision")
#    * the channel name ("ECG").
#
# For this tutorial, we will use internal files already stored in NumPy format for demonstration purposes. 
# Let's load them and plot a sample.


raw_resp = np.load('resp1_airflow.npy') # load respi
raw_ecg = np.load('ecg1.npy') # load ecg
srate = 1000. # our example signals have been recorded at 1000 Hz

times = np.arange(raw_resp.size) / srate # build time vector

linewidth = 0.8
fig, axs = plt.subplots(nrows=2, sharex=True)
ax = axs[0]
ax.plot(times, raw_resp, lw = linewidth)
ax.set_title('Raw respiratory airflow signal')
ax.set_ylabel('Amplitude (AU)')
ax.set_ylim(-1750, -1500)

ax = axs[1]
ax.plot(times, raw_ecg, lw = linewidth)
ax.set_title('Raw ECG (electrocardiogram)')
ax.set_ylabel('Amplitude (V)')
ax.set_ylim(0.012, 0.015)

ax.set_xlabel('Time (s)')
ax.set_xlim(185, 225)


##############################################################################
# 
# Analyze respiration
# -------------------
# 
# :py:func:`~physio.compute_respiration` is a high-level wrapper function that simplifies respiratory signal analysis.
# To use this function, you must provide:
#
#    * `raw_resp` : the raw respiratory signal as a NumPy array.
#    * `srate` : the sampling rate of the respiratory signal
#    * `parameter_preset` : a string specifying the type of respiratory data, which determines the set of parameters used for processing. Can be one of: `human_airflow`, `human_co2`, `human_belt`,  `rat_plethysmo`, or `rat_etisens_belt`.
#
# When called, :py:func:`~physio.compute_respiration` performs the following:
#
#    * Preprocesses the respiratory signal (returns a NumPy array: `resp`)
#    * Computes cycle-by-cycle features (returns a pd.DataFrame: `resp_cycles`)
#
# **Warning:** The orientation of the `raw_resp` trace is important (multiply it by -1 for reversing it if necessary). 
# Inspiration must point downward for `human_airflow` or `human_co2`, 
# because downward deflections are interpreted by :py:func:`~physio.compute_respiration` as inspiration. 
# For `rat_plethysmo` or `rat_etisens_belt`, the inspiration–expiration transition must point upward.




resp, resp_cycles = physio.compute_respiration(raw_resp, srate, parameter_preset = 'human_airflow') # set 'human_airflow' as preset because example resp is an airflow from humans

linewidth = 0.9
fig, ax = plt.subplots()
ax.plot(times, raw_resp, lw = linewidth, label = 'raw resp signal')
ax.plot(times, resp, lw = linewidth, label = 'preprocessed resp signal')
ax.set_ylabel('Amplitude (AU)')
ax.set_ylim(-1750, -1500)
ax.set_xlim(185, 225)
ax.set_xlabel('Time (s)')
ax.set_title('From raw to preprocessed respiratory airflow signal')
ax.legend(loc = 'upper right')

##############################################################################
# 
# Respiration cycles and features
# -------------------------------
#
# `resp_cycles` is a pd.DataFrame where each row corresponds to a respiratory cycle.
# Columns contain cycle-specific features such as duration, amplitude, volume, etc ... 
# See :ref:`sphx_glr_examples_example_02_respiration.py` for a complete description of the columns.

print(resp_cycles.shape)
print(resp_cycles.columns)

columns = ['cycle_duration', 'inspi_volume', 'expi_volume', 'total_amplitude'] # selection of some columns / metrics
colors = ['tab:blue','darkorange','green','red'] # set colors
units = ['s','AU','AU','AU'] # set units

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
fig.suptitle(f'Distribution of some metrics from N = {resp_cycles.shape[0]} detected cycles')
for ax, col, color, unit in zip(axs.flatten(), columns, colors, units):
    ax.hist(resp_cycles[col], bins=30, color=color, edgecolor = 'k') 
    ax.set_title(col)
    ax.set_xlabel(f'{col} ({unit})')
    ax.set_ylabel("Count")

fig.tight_layout()

resp_cycles


##############################################################################
# All these metrics highly depend on the quality of detection of the start points of inspiration and expiration, which can be verified by visual inspection: 

inspi_ind = resp_cycles['inspi_index'].values # get index of inspiration start points
expi_ind = resp_cycles['expi_index'].values # get index of expiration start points

fig, ax = plt.subplots()
ax.plot(times, resp)
ax.scatter(times[inspi_ind], resp[inspi_ind], color='green', label = 'inspiration start')
ax.scatter(times[expi_ind], resp[expi_ind], color='red', label = 'expiration start')
ax.set_xlim(185, 225)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude (AU)')
ax.set_ylim(-1750, -1500)
ax.set_title('Visual inspection of resp cycle detection')
ax.legend(loc = 'upper right')



##############################################################################
# 
# Analyze ECG
# -----------
#
# :py:func:`~physio.compute_ecg` is a high-level wrapper function that simplifies ECG signal analysis.
# To use this function, you must provide:
#
#    * `raw_ecg` : the raw ECG signal as a NumPy array
#    * `srate` : the sampling rate of the ECG signal
#    * `parameter_preset` : a string specifying the type of ECG data, which determines the set of parameters used for processing. Can be one of: `human_ecg` or `rat_ecg`.
#
# When called, :py:func:`~physio.compute_ecg` performs the following:
#
#    * Preprocesses the ECG signal (returns a NumPy array: `ecg`). By default, `ecg` is normalized.
#    * Detects R peaks (returns a pd.DataFrame: `ecg_peaks`)
# 
# **Warning:** The orientation of the `raw_ecg` trace is important (multiply it by -1 for reversing it if necessary). 
# R peaks must point upward for the highest probability of detection by :py:func:`~physio.compute_ecg`. 
# Sometimes the R and S peaks of the QRS complex are equivalent, so orientation does not matter, 
# because either the R or S peak will be detected and used to mark heartbeats.


ecg, ecg_peaks = physio.compute_ecg(raw_ecg, srate, parameter_preset = 'human_ecg') # set 'human_ecg' as preset because example ecg is from human

r_peak_ind = ecg_peaks['peak_index'].values # get index of R peaks

linewidth = 0.8
fig, axs = plt.subplots(nrows=2, sharex=True)
ax = axs[0]
ax.plot(times, raw_ecg, lw = linewidth)
ax.scatter(times[r_peak_ind], raw_ecg[r_peak_ind], color="#C3CB22", label = 'R peak')
ax.set_ylim(0.012, 0.015)
ax.set_ylabel('Amplitude (V)')
ax.set_title('R peak detection, plotted on raw ECG')
ax.legend(loc = 'upper right')

ax = axs[1]
ax.plot(times, ecg, lw = linewidth)
ax.scatter(times[r_peak_ind], ecg[r_peak_ind], color='#A8AE27', label = 'R peak')
ax.set_xlim(185, 225)
ax.set_ylim(-35, 50)
ax.set_ylabel('Amplitude (AU)')
ax.set_title('R peak detection, plotted on preprocessed ECG')
ax.legend(loc = 'upper right')

ax.set_xlabel('Time (s)')

##############################################################################
# 
# Heart Rate Variability metrics (time-domain)
# ---------------------------------------------
#
# :py:func:`~physio.compute_ecg_metrics`  is a high-level wrapper function that computes time-domain Heart Rate Variability (HRV) metrics from previously detected R peaks (`ecg_peaks`).
# To use this function, you must provide the previously detected R peaks, `ecg_peaks`:
#
#    * `ecg_peaks` : pd.DataFrame, output of the function :py:func:`~physio.compute_ecg`
#
# We can visualize these metrics and the RR interval distribution. See :ref:`sphx_glr_examples_example_03_ecg.py` for a complete description of the metrics.

ecg_metrics = physio.compute_ecg_metrics(ecg_peaks) # ecg_metrics = a pd.Series containing HRV time domain results

peak_times = ecg_peaks['peak_time'].values # get R peak times of detection
rri_s = np.diff(peak_times) # compute RR intervals in seconds
rri_ms = rri_s * 1000 # seconds to milliseconds

fig, ax = plt.subplots()
ax.hist(rri_ms, bins=np.arange(500, 1400, 25), edgecolor = 'k') # plot distribution of RR intervals
ax.axvline(ecg_metrics['HRV_Mean'], color='orange', label='Mean RRi') # vertical line at the mean RRi
ax.axvline(ecg_metrics['HRV_Median'], color='violet', label='Median RRi') # vertical line at the median RRi
ax.axvspan(ecg_metrics['HRV_Median'] - 2*ecg_metrics['HRV_Mad'], ecg_metrics['HRV_Median'] + 2*ecg_metrics['HRV_Mad'], color = 'orange', alpha = 0.1, label = 'Median +/- 2 * Median Absolute Deviation') # vertical span at the med + 2*mad
ax.axvspan(ecg_metrics['HRV_Mean'] - 2*ecg_metrics['HRV_SD'], ecg_metrics['HRV_Mean'] + 2*ecg_metrics['HRV_SD'], color = 'violet', alpha = 0.1, label = 'Mean +/- 2 * Standard-Deviation') # vertical span at the mean + 2*sd
ax.set_xlabel('RR interval (ms)')
ax.set_ylabel('Count')
ax.set_ylim(-5, 90)
ax.legend(loc = 'upper center', ncols = 2)
ax.set_title('Distribution or RR intervals + Heart Rate Variability metrics')
print(ecg_metrics)

##############################################################################
# 
# Heart Rate Variability metrics (frequency-domain)
# -------------------------------------------------
#
# :py:func:`~physio.compute_hrv_psd`  is a high-level wrapper function that computes frequency-domain Heart Rate Variability (HRV) metrics from previously detected R peaks (`ecg_peaks`).
# To use this function, you must provide the previously detected R peaks, `ecg_peaks`:
#
#    * `ecg_peaks` : pd.DataFrame, output of the function :py:func:`~physio.compute_ecg`
# 
# Many others parameters can be set for this function (frequency bands, size of the Welch's window, etc...). See :ref:`sphx_glr_examples_example_03_ecg.py` for more information about these.
# When called, :py:func:`~physio.compute_hrv_psd` performs the following:
#
#    * Compute an instantaneous heart rate vector from ecg peaks and compute a Fourier transform using Welch method (returns two NumPy arrays: `psd_freqs` and `psd` giving frequency vector and power vector, respectively).
#    * Compute HRV frequency metrics by getting power for each frequency band using trapezoïdal rule (returns a pd.Series: `psd_metrics`)
# 
# We can visualize these metrics and the RR interval distribution. See :ref:`sphx_glr_examples_example_03_ecg.py` for a complete description of the metrics.

frequency_bands = {'lf': (0.04, .15), 'hf' : (0.15, .4)} # set classical cutoffs of low and high frequency bands, in a dictionnary
psd_freqs, psd, psd_metrics = physio.compute_hrv_psd(ecg_peaks=ecg_peaks, frequency_bands=frequency_bands)

print(psd_metrics)
fig, ax = plt.subplots()
ax.plot(psd_freqs, psd)
colors = {'lf': '#B8860B', 'hf' : '#D2691E'}
for name, freq_band in frequency_bands.items():
    ax.axvspan(*freq_band, alpha=0.1, color=colors[name], label=f'{name} : {round(psd_metrics[name], 2)}') # plot one vertical span for each frequency band
ax.set_xlim(0, 0.6)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power Spectral Density')
ax.legend(loc = 'upper right')
ax.set_title('HRV Power Spectral Density\n-> Frequency-Domain metrics')



##############################################################################
# Cyclic deformation
# ------------------
# 
# :py:func:`~physio.deform_traces_to_cycle_template` is a tool used to deform traces to a cycle template.
# It works by stretching a trace using linear resampling based on a fixed number of points per cycle.
# This is helpful to explore if features of a signal are driven by a cyclic phenomenon like respiration.
# 
# To use this function, you must provide:
#
#    * `data` : nd.array. Axis of the time must always be 0, meaning of shape (n_times,...).
#    * `times` : np.array. Time vector of the data. Shape = (n_times,)
#    * `cycle_times` : nd.array with shape (n_cycles, n_segments + 1). Typically, for respiration, `cycle_times` is an array with 3 columns (inspi_time + expi_time + next_inspi_time) that will make deformation with 2 segments. If cycle_times is 1D, then it is converted to shape (size-1, 2). The end of every cycles must match the start of the next cycle.
#    * `points_per_cycle` : Number of phase points per cycle
#    * `segment_ratios` : None or float or list of float. None if 1 segment. Float or list of float if 2 segments. List of floats if > 2 segments. This is a ratio between 0 and 1 where cycle is divided.
#    * `output_mode` : 'stacked' / 'unstacked' / 'unstacked_full'. Format of the outputs. Stacked -> 2D matrix : cycles / points per cycle. Unstacked -> 1D matrix : flattened version of the stacked. Unstacked_full returns extra-outputs.

# Here, we deform the respiratory trace by "itself": the respiratory cycle.
# This leads to an average respiratory template. Importantly, this can be done using one or several segments inside the cycle.

# here we have 3 times per cycle so 2 segments : 
# segment 1: inspiration to expiration
# segment 2: expiration to next inspiration

cycle_times = resp_cycles[['inspi_time', 'expi_time', 'next_inspi_time']].values # get 3 timepoints per cycle, dividing each cycle in 2 segments

segment_ratios = 0.4 # ratio between 0 and 1 where cycle is divided into 2 segments
deformed_resp_2seg = physio.deform_traces_to_cycle_template(resp, # resp trace to deform
                                                            times, # time vector of resp trace to deform
                                                             cycle_times,  # times of resp cycles, used to strech
                                                             points_per_cycle=40,  # number of points per cycle used for linear resampling
                                                             segment_ratios=segment_ratios, # ratio between 0 and 1 where cycle is divided into 2 segments
                                                             output_mode='stacked' # choose a stacked version of the returned matrix (see docstrings)
                                                             )
print(deformed_resp_2seg.shape, cycle_times.shape)

# here we have 2 times per cycle so 1 segment:
# segment 1: inspiration to next inspiration

cycle_times = resp_cycles[['inspi_time', 'next_inspi_time']].values # get 2 timepoints per cycle, dividing each cycle in 1 segment (so not dividing)
deformed_resp_1seg = physio.deform_traces_to_cycle_template(resp, # resp trace to deform
                                                            times, # time vector of resp trace to deform
                                                             cycle_times,  # times of resp cycles, used to strech
                                                             points_per_cycle=40,  # number of points per cycle used for linear resampling
                                                             segment_ratios=None, # ratio between 0 and 1 where cycle is divided into 2 segments. None in this case because 1 segment
                                                             output_mode='stacked' # choose a stacked version of the returned matrix (see docstrings)
                                                             )
print(deformed_resp_1seg.shape, cycle_times.shape)

fig, axs = plt.subplots(ncols=2, sharey=True, figsize = (8,4))

ax = axs[0]
physio.plot_cyclic_deformation(deformed_resp_2seg, segment_ratios=0.4, two_cycles=False, ax=ax) # using function physio.plot_cyclic_deformation to plot
ax.set_title('Deformation: 2 segments')
ax.set_ylabel('Resp Amplitude (AU)')

ax = axs[1]
physio.plot_cyclic_deformation(deformed_resp_1seg, segment_ratios=None, two_cycles=False, ax=ax) # using function physio.plot_cyclic_deformation to plot
ax.set_title('Deformation: 1 segment')

for ax in axs:
    ax.set_xlabel('Points per cycle (= respiratory phase)')


##############################################################################
# 
# Cyclic deformation on ECG
# -------------------------
# 
# Let's use the same :py:func:`~physio.deform_traces_to_cycle_template` tool for ECG trace.
# We can use a simple time vector: the R peak times. In this case, the ECG trace is converted as a 1-segment.


cycle_times = ecg_peaks['peak_time'].values # one cycle = R to R -> one segment by R to R period
deformed_ecg = physio.deform_traces_to_cycle_template(ecg, # ecg trace to deform
                                                      times, # time vector of ecg trace to deform
                                                      cycle_times, # times of ecg cycles = R to R periods, used to strech
                                                      points_per_cycle=300, # number of points per cycle used for linear resampling (more than for resp because ecg trace has faster dynamics)
                                                      segment_ratios=None, # 1 segment so None
                                                      output_mode='stacked'
                                                      )
print(deformed_ecg.shape, cycle_times.shape)

fig, ax = plt.subplots()
physio.plot_cyclic_deformation(deformed_ecg, two_cycles=True, ax=ax) # set two_cycles = True to see a concatenation of two cyclical deformations, so that we can see the whole PQRST complex
ax.set_title('Two ECG cyclical deformations concatenated:\nnormalized view of the whole PQRST complex')
ax.set_ylabel('Amplitude (AU)')
ax.set_xlabel('Points per cycle * 2')

##############################################################################
# 

