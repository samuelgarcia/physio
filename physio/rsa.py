import numpy as np
import pandas as pd

from .ecg import compute_instantaneous_rate
from .cyclic_deformation import deform_traces_to_cycle_template


def compute_rsa(resp_cycles, ecg_peaks, srate=100., units='bpm', limits=None, two_segment=True, points_per_cycle=50):
    """
    RSA = Respiratory Sinus Arrhythmia

    Compute the RSA with the cyclic way : 
      * compute instanteneous heart rate
      * on resp cycle basis compute peak-to-trough

      Also compute the cyclic deformation of the instantaneous heart rate

    Parameters
    ----------
    resp_cycles
    
    ecg_peaks
    
    srate
    100 is safe for both animal and human for human 10 is also OK.

    units

    limits

    two_segment

    points_per_cycle

    Returns
    -------
    rsa_cycles

    cyclic_cardiac_rate
    
    """
    
    

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
    

    rsa_cycles = pd.DataFrame(index=resp_cycles.index)

    n = resp_cycles.shape[0]
    rsa_cycles['peak_index'] = pd.Series(np.zeros(n), dtype='int64')
    rsa_cycles['trough_index'] = pd.Series(np.zeros(n), dtype='int64')

    columns=['peak_time', 'trough_time',
             'peak_value', 'trough_value',
             'rising_amplitude', 'decay_amplitude',
             'rising_duration', 'decay_duration',
             'rising_slope', 'decay_slope',
             ]
    for col in columns:
        rsa_cycles[col] = pd.Series(dtype='float64')
    
    for c, cycle in resp_cycles.iterrows():
        t0, t1 = cycle['inspi_time'], cycle['next_inspi_time']
        i0, i1 = int(t0 * srate), int(t1 * srate)
        chunk = instantaneous_cardiac_rate[i0:i1]

        ind_max = np.argmax(chunk)
        ind_min = np.argmin(chunk[ind_max:]) + ind_max

        rsa_cycles.at[c, 'peak_index'] = i0 + ind_max
        rsa_cycles.at[c, 'trough_index'] = i0 + ind_min
        rsa_cycles.at[c, 'peak_time'] = t0 + ind_max / srate
        rsa_cycles.at[c, 'trough_time'] = t0 + ind_min / srate

    rsa_cycles['peak_value'] = instantaneous_cardiac_rate[rsa_cycles['peak_index'].values]
    rsa_cycles['trough_value'] = instantaneous_cardiac_rate[rsa_cycles['trough_index'].values]

    rsa_cycles['decay_amplitude'] = rsa_cycles['peak_value'] - rsa_cycles['trough_value']
    rsa_cycles['rising_amplitude'].values[1:] = rsa_cycles['peak_value'].values[1:] - rsa_cycles['trough_value'].values[:-1]

    rsa_cycles['rising_duration'].values[1:] = rsa_cycles['peak_time'].values[1:] - rsa_cycles['trough_time'].values[:-1]
    rsa_cycles['decay_duration'] = rsa_cycles['trough_time'] - rsa_cycles['peak_time']

    rsa_cycles['rising_slope'] = rsa_cycles['rising_amplitude'] / rsa_cycles['rising_duration']
    rsa_cycles['decay_slope'] = rsa_cycles['decay_amplitude'] / rsa_cycles['decay_duration']

    
    return rsa_cycles, cyclic_cardiac_rate
