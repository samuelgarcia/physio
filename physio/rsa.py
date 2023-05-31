import numpy as np
import pandas as pd

from .ecg import compute_instantaneous_rate
from .cyclic_deformation import deform_traces_to_cycle_template


def compute_rsa(resp_cycles, ecg_peaks, srate=50., units='bpm', two_segment=True, points_per_cycle=50):
    """
    

    Parameters
    ----------
    resp_cycles
    
    ecg_peaks
    
    srate

    units

    two_segment

    points_per_cycle

    Returns
    -------
    rsa_cycles

    cyclic_cardiac_rate
    
    """
    
    

    duration_s = max(resp_cycles['next_inspi_time'].values[-1], ecg_peaks['peak_time'].values[-1])
    print(duration_s)

    # times = np.arange(0,  duration_s + 1 / srate, 1 / srate)
    times = np.arange(0,  duration_s, 1 / srate)
    instantaneous_cardiac_rate = compute_instantaneous_rate(ecg_peaks, times, limits=None,
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
    

    rsa_cycles = pd.DataFrame(index=resp_cycles.index, columns=['amplitude', 'peak_value', 'trough_value'])

    rsa_cycles['amplitude'] = np.ptp(cyclic_cardiac_rate, axis=1)
    rsa_cycles['peak_value'] = np.max(cyclic_cardiac_rate, axis=1)
    rsa_cycles['trough_value'] = np.min(cyclic_cardiac_rate, axis=1)

    
    return rsa_cycles, cyclic_cardiac_rate
