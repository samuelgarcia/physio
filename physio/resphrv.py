import numpy as np
import pandas as pd

from .ecg import compute_instantaneous_rate
from .cyclic_deformation import deform_traces_to_cycle_template

import warnings


def compute_resphrv(resp_cycles, ecg_peaks, srate=100., units='bpm', limits=None, two_segment=True, points_per_cycle=50):
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
        Sampling rate used for interpolation to get an instantaneous heart rate vector. 
        100 is safe for both animal and human. For human 10 also works.
    units : str
        bpm / s / ms / Hz
    limits : list or None
        Limits for removing outliers. To set according to the units parameter. Ex : [30, 200] to remove RR intervals out of this range set in bpm.
    two_segment : bool
        True or False
    points_per_cycle : int

    Returns
    -------
    resphrv_cycles : pd.DataFrame
        Cycle-by-cycle features of Heart Rate dynamics. Ex : decay_amplitude gives the by-cycle peak-to-trough amplitude.
    cyclic_cardiac_rate : nd.array
        2D Matrix (respiratory cycle * respiratory phase) with instantaneous heart rate at each resp cycle and phase point.
    """
    

    t0, t1 = resp_cycles['inspi_time'].values[0], resp_cycles['next_inspi_time'].values[-1]

    times = np.arange(t0,  t1 + 1 / srate, 1 / srate)
    i0_ref = int(t0 * srate)

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
    resphrv_cycles['peak_index'] = pd.Series(np.zeros(n), dtype='int64')
    resphrv_cycles['trough_index'] = pd.Series(np.zeros(n), dtype='int64')

    columns=['peak_time', 'trough_time',
             'peak_value', 'trough_value',
             'rising_amplitude', 'decay_amplitude',
             'rising_duration', 'decay_duration',
             'rising_slope', 'decay_slope',
             ]
    for col in columns:
        resphrv_cycles[col] = pd.Series(dtype='float64')
    
    for c, cycle in resp_cycles.iterrows():
        t0, t1 = cycle['inspi_time'], cycle['next_inspi_time']
        i0, i1 = int(t0 * srate), int(t1 * srate)
        i0 -= i0_ref
        i1 -= i0_ref
        chunk = instantaneous_cardiac_rate[i0:i1]

        ind_max = np.argmax(chunk)
        ind_min = np.argmin(chunk[ind_max:]) + ind_max

        resphrv_cycles.at[c, 'peak_index'] = i0 + ind_max
        resphrv_cycles.at[c, 'trough_index'] = i0 + ind_min
        resphrv_cycles.at[c, 'peak_time'] = t0 + ind_max / srate
        resphrv_cycles.at[c, 'trough_time'] = t0 + ind_min / srate

    resphrv_cycles['peak_value'] = instantaneous_cardiac_rate[resphrv_cycles['peak_index'].values]
    resphrv_cycles['trough_value'] = instantaneous_cardiac_rate[resphrv_cycles['trough_index'].values]

    resphrv_cycles['decay_amplitude'] = resphrv_cycles['peak_value'] - resphrv_cycles['trough_value']
    resphrv_cycles['rising_amplitude'].values[1:] = resphrv_cycles['peak_value'].values[1:] - resphrv_cycles['trough_value'].values[:-1]

    resphrv_cycles['rising_duration'].values[1:] = resphrv_cycles['peak_time'].values[1:] - resphrv_cycles['trough_time'].values[:-1]
    resphrv_cycles['decay_duration'] = resphrv_cycles['trough_time'] - resphrv_cycles['peak_time']

    resphrv_cycles['rising_slope'] = resphrv_cycles['rising_amplitude'] / resphrv_cycles['rising_duration']
    resphrv_cycles['decay_slope'] = resphrv_cycles['decay_amplitude'] / resphrv_cycles['decay_duration']

    
    return resphrv_cycles, cyclic_cardiac_rate



def compute_rsa(*args, **kwargs):
    warnings.warn('compute_rsa() has been renamed to compute_resphrv(). compute_rsa() will be removed')
    return compute_resphrv(*args, **kwargs)