import numpy as np
import pandas as pd

from .ecg import compute_instantaneous_rate
from .cyclic_deformation import deform_traces_to_cycle_template

import warnings


def compute_resphrv(resp_cycles, ecg_peaks, srate=100., units='bpm', limits=None, two_segment=True, points_per_cycle=50, return_cyclic_cardiac_rate=True):
    """
    RSA = Respiratory Sinus Arrhythmia (or Respiratory Heart Rate Variability / RespHRV)

    Compute the RSA cycle-by-cycle : 
      * compute instantaneous heart rate
      * on resp cycle basis compute peak-to-trough

    Also compute the cyclic deformation of the instantaneous heart rate

    Parameters
    ----------

    resp_cycles : pd.DataFrame
        DataFrame of detected respiratory cycles
    ecg_peaks : pd.DataFrame
        DataFrame of detected ecg R peaks
    srate : int or float
        Sampling rate used for interpolation to get an instantaneous heart rate vector, to compute cyclic_cardiac_rate. 
        100 is safe for both animal and human. For human 10 also works.
    units : str
        bpm / Hz
    limits : list or None
        Limits for removing outliers. To set according to the units parameter. Ex : [30, 200] to remove heart rates (in bpm) out of this range.
    two_segment : bool
        True or False (default = True). Deform instantaneous heart rate by respiratory phase using one segment (inspi_time to next_inspi_time) or two segments (inspi_time to expi_time and expi_time to next_inspi_time), to compute cyclic_cardiac_rate.
    points_per_cycle : int
        Number of respiratory phase points per cycle, used in deform_traces_to_cycle_template() to build cyclic_cardiac_rate matrix
    return_cyclic_cardiac_rate : bool
        If True, returns both outputs (resphrv_cycles and cyclic_cardiac_rate), else computes and returns only resphrv_cycles and not cyclic_cardiac_rate

    Returns
    -------
    resphrv_cycles : pd.DataFrame
        Cycle-by-cycle features of Heart Rate dynamics. Ex : decay_amplitude gives the by-cycle peak-to-trough amplitude.
    cyclic_cardiac_rate : nd.array
        2D Matrix (respiratory cycle * respiratory phase) with instantaneous heart rate at each resp cycle and phase point.
    """
    
    assert units in ('Hz', 'bpm'), "For RespHRV, units must be bpm or Hz"


    if return_cyclic_cardiac_rate:
        duration_s = resp_cycles['next_inspi_time'].values[-1]

        times = np.arange(0,  duration_s + 1 / srate, 1 / srate)
        instantaneous_cardiac_rate = compute_instantaneous_rate(ecg_peaks, times, limits=limits,
                                                                units=units, interpolation_kind='linear')    
        
        if two_segment:
            cycle_times = resp_cycles[['inspi_time', 'expi_time','next_inspi_time']].values
            inspi_ratio = np.mean((cycle_times[:, 1] - cycle_times[:, 0]) / (cycle_times[:, 2] - cycle_times[:, 0]))
            segment_ratios = [inspi_ratio]
        else:
            cycle_times = resp_cycles[['inspi_time', 'next_inspi_time']].values
            segment_ratios = None

        cyclic_cardiac_rate = deform_traces_to_cycle_template(instantaneous_cardiac_rate, times, cycle_times,
                                                        points_per_cycle=points_per_cycle, segment_ratios=segment_ratios)
    

    resphrv_cycles = pd.DataFrame(index=resp_cycles.index)

    n = resp_cycles.shape[0]

    columns=['peak_time', 'trough_time',
             'peak_value', 'trough_value',
             'min_max_amplitude','relative_min_max_amplitude',
             'rising_amplitude', 'relative_rising_amplitude' ,'decay_amplitude', 'relative_decay_amplitude',
             'rising_duration', 'decay_duration',
             'rising_slope', 'decay_slope',
             ]
    for col in columns:
        resphrv_cycles[col] = pd.Series(dtype='float64')

    ecg_peak_times = ecg_peaks['peak_time'].values
    delta = np.diff(ecg_peak_times)
    ecg_peak_times = ecg_peak_times[:-1]

    if units == 'Hz':
        hrate = 1.  / delta
    elif units == 'bpm':
        hrate = 60.  / delta
    else:
        raise ValueError(f'Bad units {units}')

    if not limits is None:
        mask = (hrate > limits[0]) & (hrate < limits[1])
        hrate = hrate[mask]
        ecg_peak_times = ecg_peak_times[mask]
    
    for c, cycle in resp_cycles.iterrows():
        t0, t1 = cycle['inspi_time'], cycle['next_inspi_time']

        ind0, ind1 =  np.searchsorted(ecg_peak_times, [t0, t1])
        if ind0 == ind1:
            continue

        ind_max = np.argmax(hrate[ind0:ind1]) + ind0
        resphrv_cycles.at[c, 'peak_time'] = ecg_peak_times[ind_max]
        resphrv_cycles.at[c, 'peak_value'] = hrate[ind_max]

        if ind1 - ind0 >= 2:
            max_, min_ = np.max(hrate[ind0:ind1]), np.min(hrate[ind0:ind1])
            ptp = max_ - min_
            relative_ptp = ptp / (max_ + min_)
            resphrv_cycles.at[c, 'min_max_amplitude'] = ptp
            resphrv_cycles.at[c, 'relative_min_max_amplitude'] = relative_ptp


    for c, cycle in resp_cycles.iloc[:-1].iterrows():
        t0 = resphrv_cycles.loc[c, 'peak_time']
        t1 = resphrv_cycles.loc[c+1, 'peak_time']

        if np.isnan(t0) or np.isnan(t1):
            continue

        ind0, ind1 = np.searchsorted(ecg_peak_times, [t0, t1])
        ind0 += 1
        if ind0+1 >= ind1:
            continue

        ind_min = np.argmin(hrate[ind0:ind1]) + ind0
        resphrv_cycles.at[c, 'trough_time'] = ecg_peak_times[ind_min]
        resphrv_cycles.at[c, 'trough_value'] = hrate[ind_min]


    resphrv_cycles['decay_amplitude'] = resphrv_cycles['peak_value'] - resphrv_cycles['trough_value']
    resphrv_cycles['relative_decay_amplitude'] = resphrv_cycles['decay_amplitude'] / (resphrv_cycles['peak_value'] + resphrv_cycles['trough_value'])
    resphrv_cycles.loc[resphrv_cycles.index[1:] ,'rising_amplitude'] = resphrv_cycles['peak_value'].values[1:] - resphrv_cycles['trough_value'].values[:-1]
    resphrv_cycles.loc[resphrv_cycles.index[1:] ,'relative_rising_amplitude'] = resphrv_cycles.loc[resphrv_cycles.index[1:] ,'rising_amplitude'] / (resphrv_cycles['peak_value'].values[1:] + resphrv_cycles['trough_value'].values[:-1])

    mask = resphrv_cycles['decay_amplitude'] < 0
    resphrv_cycles.loc[mask,'decay_amplitude'] = np.nan
    resphrv_cycles.loc[mask,'relative_decay_amplitude'] = np.nan
    mask = resphrv_cycles['rising_amplitude'] < 0
    resphrv_cycles.loc[mask,'rising_amplitude'] = np.nan
    resphrv_cycles.loc[mask,'relative_rising_amplitude'] = np.nan

    resphrv_cycles.loc[resphrv_cycles.index[1:] ,'rising_duration'] = resphrv_cycles['peak_time'].values[1:] - resphrv_cycles['trough_time'].values[:-1]
    resphrv_cycles['decay_duration'] = resphrv_cycles['trough_time'] - resphrv_cycles['peak_time']

    resphrv_cycles['rising_slope'] = resphrv_cycles['rising_amplitude'] / resphrv_cycles['rising_duration']
    resphrv_cycles['decay_slope'] = resphrv_cycles['decay_amplitude'] / resphrv_cycles['decay_duration']

    if return_cyclic_cardiac_rate:
        return resphrv_cycles, cyclic_cardiac_rate
    else:
        return resphrv_cycles
    
def compute_resphrv_rate_period(resp_cycles, ecg_peaks, srate=100., bpm_limits=None, two_segment=True, points_per_cycle=50, return_cyclic_cardiac_rate=True, return_cyclic_cardiac_period = True):
    """
    RespHRV = Respiratory Heart Rate Variability (or RespHRV, ex. Respiratory Sinus Arrhythmia (RSA))

    Compute the RespHRV/RSA cycle-by-cycle using as a rate unit the beats per minute (bpm) and as a time period unit the milliseconds (ms) and does it : 
      * On a respiratory cycle basis, i.e. compute RespHRV using peak-to-trough framework
      * Can compute instantaneous heart rate in bpm and deform it cyclically based on respiratory time basis to get it at each resp cycle and phase point
      * Can compute instantaneous heart period in ms and deform it cyclically based on respiratory time basis to get it at each resp cycle and phase point

    Parameters
    ----------

    resp_cycles : pd.DataFrame
        DataFrame of detected respiratory cycles
    ecg_peaks : pd.DataFrame
        DataFrame of detected ecg R peaks
    srate : int or float
        Sampling rate used for interpolation to get an instantaneous heart rate vector, to compute cyclic_cardiac_rate. 
        100 is safe for both animal and human. For human 10 also works.
    bpm_limits : list or None
        Limits for removing outliers. To set according in beats per minute as a unit. Ex : [30, 200] to remove heart rates in bpm out of this range.
    two_segment : bool
        True or False (default = True). Deform instantaneous heart rate by respiratory phase using one segment (inspi_time to next_inspi_time) or two segments (inspi_time to expi_time and expi_time to next_inspi_time), to compute cyclic_cardiac_rate.
    points_per_cycle : int
        Number of respiratory phase points per cycle, used in deform_traces_to_cycle_template() to build cyclic_cardiac_rate matrix
    return_cyclic_cardiac_rate : bool
        If True, returns as additional output cyclic_cardiac_rate, i.e. the heart rate in bpm according to the respiratory phase of each resp cycle
    return_cyclic_cardiac_period : bool
        If True, returns as additional output cyclic_cardiac_period, i.e. the heart period in ms according to the respiratory phase of each resp cycle

    Returns
    -------
    resphrv_cycles : pd.DataFrame
        Cycle-by-cycle features of Heart Rate dynamics. Ex : decay_amplitude gives the by-cycle peak-to-trough amplitude.
    cyclic_cardiac_rate : nd.array
        2D Matrix (respiratory cycle * respiratory phase) with instantaneous heart rate in bpm at each resp cycle and phase point.
    cyclic_cardiac_period : nd.array
        2D Matrix (respiratory cycle * respiratory phase) with instantaneous heart period in ms at each resp cycle and phase point.
    """
    # Force units to be in beats per minute for rate and in milliseconds for periods
    rate_units = 'bpm'
    period_units = 'ms'

    rate_limits = bpm_limits

    if not rate_limits is None:
        period_limits = (60000 / rate_limits[1], 60000 / rate_limits[0])
    else:
        period_limits = None

    if return_cyclic_cardiac_rate:
        duration_s = resp_cycles['next_inspi_time'].values[-1]

        times = np.arange(0,  duration_s + 1 / srate, 1 / srate)
        instantaneous_cardiac_rate = compute_instantaneous_rate(ecg_peaks, times, limits=rate_limits, units=rate_units, interpolation_kind='linear')    
        
        if two_segment:
            cycle_times = resp_cycles[['inspi_time', 'expi_time','next_inspi_time']].values
            inspi_ratio = np.mean((cycle_times[:, 1] - cycle_times[:, 0]) / (cycle_times[:, 2] - cycle_times[:, 0]))
            segment_ratios = [inspi_ratio]
        else:
            cycle_times = resp_cycles[['inspi_time', 'next_inspi_time']].values
            segment_ratios = None

        cyclic_cardiac_rate = deform_traces_to_cycle_template(instantaneous_cardiac_rate, times, cycle_times, points_per_cycle=points_per_cycle, segment_ratios=segment_ratios)

    if return_cyclic_cardiac_period:
        duration_s = resp_cycles['next_inspi_time'].values[-1]

        times = np.arange(0,  duration_s + 1 / srate, 1 / srate)
        instantaneous_cardiac_period = compute_instantaneous_rate(ecg_peaks, times, limits=period_limits, units=period_units, interpolation_kind='linear')    
        
        if two_segment:
            cycle_times = resp_cycles[['inspi_time', 'expi_time','next_inspi_time']].values
            inspi_ratio = np.mean((cycle_times[:, 1] - cycle_times[:, 0]) / (cycle_times[:, 2] - cycle_times[:, 0]))
            segment_ratios = [inspi_ratio]
        else:
            cycle_times = resp_cycles[['inspi_time', 'next_inspi_time']].values
            segment_ratios = None

        cyclic_cardiac_period = deform_traces_to_cycle_template(instantaneous_cardiac_period, times, cycle_times, points_per_cycle=points_per_cycle, segment_ratios=segment_ratios)

    resphrv_cycles = pd.DataFrame(index=resp_cycles.index)

    n = resp_cycles.shape[0]

    columns_rate=['peak_time_bpm', 'trough_time_bpm',
             'peak_value_bpm', 'trough_value_bpm',
             'min_max_amplitude_bpm','relative_min_max_amplitude_bpm',
             'rising_amplitude_bpm', 'relative_rising_amplitude_bpm' ,'decay_amplitude_bpm', 'relative_decay_amplitude_bpm',
             'rising_duration_bpm', 'decay_duration_bpm',
             'rising_slope_bpm', 'decay_slope_bpm',
             ]
    columns_period=['trough_time_ms', 'peak_time_ms',
             'trough_value_ms', 'peak_value_ms',
             'min_max_amplitude_ms','relative_min_max_amplitude_ms',
             'decay_amplitude_ms', 'relative_decay_amplitude_ms' ,'rising_amplitude_ms', 'relative_rising_amplitude_ms',
             'decay_duration_ms', 'rising_duration_ms',
             'decay_slope_ms', 'rising_slope_ms',
             ]
    columns = columns_rate + columns_period
    
    for col in columns:
        resphrv_cycles[col] = pd.Series(dtype='float64')

    ecg_peak_times = ecg_peaks['peak_time'].values
    delta_s = np.diff(ecg_peak_times)
    ecg_peak_times = ecg_peak_times[:-1]

    hrate = 60.  / delta_s
    hperiod = 1000. * delta_s
    
    if not bpm_limits is None:
        mask = (hrate > bpm_limits[0]) & (hrate < bpm_limits[1])
        hrate = hrate[mask]
        hperiod = hperiod[mask]
        ecg_peak_times = ecg_peak_times[mask]

    for c, cycle in resp_cycles.iterrows():
        t0, t1 = cycle['inspi_time'], cycle['next_inspi_time']

        ind0, ind1 =  np.searchsorted(ecg_peak_times, [t0, t1])
        if ind0 == ind1:
            continue

        ind_max_rate = np.argmax(hrate[ind0:ind1]) + ind0
        resphrv_cycles.at[c, 'peak_time_bpm'] = ecg_peak_times[ind_max_rate]
        resphrv_cycles.at[c, 'peak_value_bpm'] = hrate[ind_max_rate]
        resphrv_cycles.at[c, 'trough_time_ms'] = ecg_peak_times[ind_max_rate]
        resphrv_cycles.at[c, 'trough_value_ms'] = hperiod[ind_max_rate]

        if ind1 - ind0 >= 2: # check if enough ecg peaks detected during the resp cycle to compute resphrv min max
            max_rate, min_rate = np.max(hrate[ind0:ind1]), np.min(hrate[ind0:ind1])
            ptp_rate = max_rate - min_rate
            relative_ptp_rate = ptp_rate / (max_rate + min_rate)
            resphrv_cycles.at[c, 'min_max_amplitude_bpm'] = ptp_rate
            resphrv_cycles.at[c, 'relative_min_max_amplitude_bpm'] = relative_ptp_rate

            max_period, min_period = np.max(hperiod[ind0:ind1]), np.min(hperiod[ind0:ind1])
            ptp_period = max_period - min_period
            relative_ptp_period = ptp_period / (max_period + min_period)
            resphrv_cycles.at[c, 'min_max_amplitude_ms'] = ptp_period
            resphrv_cycles.at[c, 'relative_min_max_amplitude_ms'] = relative_ptp_period


    for c, cycle in resp_cycles.iloc[:-1].iterrows():
        t0 = resphrv_cycles.loc[c, 'peak_time_bpm']
        t1 = resphrv_cycles.loc[c+1, 'peak_time_bpm']

        if np.isnan(t0) or np.isnan(t1):
            continue

        ind0, ind1 = np.searchsorted(ecg_peak_times, [t0, t1])
        ind0 += 1
        if ind0+1 >= ind1:
            continue

        ind_min_rate = np.argmin(hrate[ind0:ind1]) + ind0
        resphrv_cycles.at[c, 'trough_time_bpm'] = ecg_peak_times[ind_min_rate]
        resphrv_cycles.at[c, 'trough_value_bpm'] = hrate[ind_min_rate]
        resphrv_cycles.at[c, 'peak_time_ms'] = ecg_peak_times[ind_min_rate]
        resphrv_cycles.at[c, 'peak_value_ms'] = hperiod[ind_min_rate]


    resphrv_cycles['decay_amplitude_bpm'] = resphrv_cycles['peak_value_bpm'] - resphrv_cycles['trough_value_bpm']
    resphrv_cycles['rising_amplitude_ms'] = abs(resphrv_cycles['trough_value_ms'] - resphrv_cycles['peak_value_ms'])
    resphrv_cycles['relative_decay_amplitude_bpm'] = resphrv_cycles['decay_amplitude_bpm'] / (resphrv_cycles['peak_value_bpm'] + resphrv_cycles['trough_value_bpm'])
    resphrv_cycles['relative_rising_amplitude_ms'] = resphrv_cycles['rising_amplitude_ms'] / (resphrv_cycles['peak_value_ms'] + resphrv_cycles['peak_value_ms'])
    resphrv_cycles.loc[resphrv_cycles.index[1:] ,'rising_amplitude_bpm'] = resphrv_cycles['peak_value_bpm'].values[1:] - resphrv_cycles['trough_value_bpm'].values[:-1]
    resphrv_cycles.loc[resphrv_cycles.index[1:] ,'decay_amplitude_ms'] = abs(resphrv_cycles['trough_value_ms'].values[1:] - resphrv_cycles['peak_value_ms'].values[:-1])
    resphrv_cycles.loc[resphrv_cycles.index[1:] ,'relative_rising_amplitude_bpm'] = resphrv_cycles.loc[resphrv_cycles.index[1:] ,'rising_amplitude_bpm'] / (resphrv_cycles['peak_value_bpm'].values[1:] + resphrv_cycles['trough_value_bpm'].values[:-1])
    resphrv_cycles.loc[resphrv_cycles.index[1:] ,'relative_decay_amplitude_ms'] = resphrv_cycles.loc[resphrv_cycles.index[1:] ,'decay_amplitude_ms'] / (resphrv_cycles['trough_value_ms'].values[1:] + resphrv_cycles['peak_value_ms'].values[:-1])

    mask = resphrv_cycles['decay_amplitude_bpm'] < 0
    resphrv_cycles.loc[mask,'decay_amplitude_bpm'] = np.nan
    resphrv_cycles.loc[mask,'relative_decay_amplitude_bpm'] = np.nan
    resphrv_cycles.loc[mask,'rising_amplitude_ms'] = np.nan
    resphrv_cycles.loc[mask,'relative_rising_amplitude_ms'] = np.nan
    mask = resphrv_cycles['rising_amplitude_bpm'] < 0
    resphrv_cycles.loc[mask,'rising_amplitude_bpm'] = np.nan
    resphrv_cycles.loc[mask,'relative_rising_amplitude_bpm'] = np.nan
    resphrv_cycles.loc[mask,'decay_amplitude_ms'] = np.nan
    resphrv_cycles.loc[mask,'relative_decay_amplitude_ms'] = np.nan

    resphrv_cycles.loc[resphrv_cycles.index[1:] ,'rising_duration_bpm'] = resphrv_cycles['peak_time_bpm'].values[1:] - resphrv_cycles['trough_time_bpm'].values[:-1]
    resphrv_cycles.loc[resphrv_cycles.index[1:] ,'decay_duration_ms'] = resphrv_cycles['trough_time_ms'].values[1:] - resphrv_cycles['peak_time_ms'].values[:-1]
    resphrv_cycles['decay_duration_bpm'] = resphrv_cycles['trough_time_bpm'] - resphrv_cycles['peak_time_bpm']
    resphrv_cycles['rising_duration_ms'] = resphrv_cycles['peak_time_ms'] - resphrv_cycles['trough_time_ms']

    resphrv_cycles['rising_slope_bpm'] = resphrv_cycles['rising_amplitude_bpm'] / resphrv_cycles['rising_duration_bpm']
    resphrv_cycles['decay_slope_ms'] = resphrv_cycles['decay_amplitude_ms'] / resphrv_cycles['decay_duration_ms']
    resphrv_cycles['decay_slope_bpm'] = resphrv_cycles['decay_amplitude_bpm'] / resphrv_cycles['decay_duration_bpm']
    resphrv_cycles['rising_slope_ms'] = resphrv_cycles['rising_amplitude_ms'] / resphrv_cycles['rising_duration_ms']

    resphrv_cycles['average_rate_bpm'] = (resphrv_cycles['peak_value_bpm'] + resphrv_cycles['trough_value_bpm']) / 2
    resphrv_cycles['average_period_ms'] = (resphrv_cycles['peak_value_ms'] + resphrv_cycles['trough_value_ms']) / 2

    if return_cyclic_cardiac_rate and return_cyclic_cardiac_period:
        return resphrv_cycles, cyclic_cardiac_rate, cyclic_cardiac_period
    elif not return_cyclic_cardiac_rate and return_cyclic_cardiac_period:
        return resphrv_cycles, cyclic_cardiac_period
    elif return_cyclic_cardiac_rate and not return_cyclic_cardiac_period:
        return resphrv_cycles, cyclic_cardiac_rate
    else:
        return resphrv_cycles



def compute_rsa(*args, **kwargs):
    warnings.warn('compute_rsa() has been renamed to compute_resphrv(). compute_rsa() will be removed')
    return compute_resphrv(*args, **kwargs)