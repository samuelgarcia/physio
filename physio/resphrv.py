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
             'min_max_amplitude',
             'rising_amplitude', 'decay_amplitude',
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
            resphrv_cycles.at[c, 'min_max_amplitude'] = np.ptp(hrate[ind0:ind1])


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
    resphrv_cycles['rising_amplitude'].values[1:] = resphrv_cycles['peak_value'].values[1:] - resphrv_cycles['trough_value'].values[:-1]

    mask = resphrv_cycles['decay_amplitude'] < 0
    resphrv_cycles.loc[mask,'decay_amplitude'] = np.nan
    mask = resphrv_cycles['rising_amplitude'] < 0
    resphrv_cycles.loc[mask,'rising_amplitude'] = np.nan

    resphrv_cycles['rising_duration'].values[1:] = resphrv_cycles['peak_time'].values[1:] - resphrv_cycles['trough_time'].values[:-1]
    resphrv_cycles['decay_duration'] = resphrv_cycles['trough_time'] - resphrv_cycles['peak_time']

    resphrv_cycles['rising_slope'] = resphrv_cycles['rising_amplitude'] / resphrv_cycles['rising_duration']
    resphrv_cycles['decay_slope'] = resphrv_cycles['decay_amplitude'] / resphrv_cycles['decay_duration']

    if return_cyclic_cardiac_rate:
        return resphrv_cycles, cyclic_cardiac_rate
    else:
        return resphrv_cycles



def compute_rsa(*args, **kwargs):
    warnings.warn('compute_rsa() has been renamed to compute_resphrv(). compute_rsa() will be removed')
    return compute_resphrv(*args, **kwargs)