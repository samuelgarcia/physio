import numpy as np
import pandas as pd

import scipy.interpolate

from .tools import detect_peak, compute_median_mad
from .preprocess import preprocess



def compute_ecg(raw_ecg, srate):
    """
    Function for ECG that:
      * preprocess the ECG
      * detect R peaks
      * apply some cleaning to remove too small ECG interval

    Parameters
    ----------
    raw_ecg: np.array
        Raw traces of ECG signal
    srate: float
        Sampling rate
    Returns
    -------
    clean_ecg: np.array
        preprocess and normalized ecg traces
    ecg_R_peaks: np.array
        Indices of R peaks
    """
    clean_ecg = preprocess(raw_ecg, srate, band=[5., 45.], ftype='bessel', order=5, normalize=True)
    
    # TODO estimation du seuil
    
    raw_ecg_peak = detect_peak(clean_ecg, srate, thresh=5, exclude_sweep_ms=4.0)
    
    ecg_R_peaks = clean_ecg_peak(clean_ecg, srate, raw_ecg_peak)
    
    return clean_ecg, ecg_R_peaks




def clean_ecg_peak(ecg, srate, raw_peak_inds, min_interval_ms=400.):
    """
    Clean peak with ultra simple idea: remove short interval.


    Parameters
    ----------
    ecg: np.array
        preprocess traces of ECG signal
    srate: float
        Sampling rate
    raw_peak_inds: np.array
        Array of peaks indices to be cleaned
    min_interval_ms: float (dfault 400ms)
        Minimum interval for cleaning
    Returns
    -------
    peak_inds: np.array
        Cleaned array of peaks indices 
    """
    
    # when two peaks are too close :  remove the smaller peaks in amplitude
    peak_ms = (raw_peak_inds / srate * 1000.)
    bad_peak, = np.nonzero(np.diff(peak_ms) < min_interval_ms)
    bad_ampl  = ecg[raw_peak_inds[bad_peak]]
    bad_ampl_next  = ecg[raw_peak_inds[bad_peak + 1]]
    bad_peak +=(bad_ampl > bad_ampl_next).astype(int)
    
    keep = np.ones(raw_peak_inds.size, dtype='bool')
    keep[bad_peak] = False
    peak_inds = raw_peak_inds[keep]
    
    return peak_inds




def compute_ecg_metrics(ecg_R_peaks, srate, min_interval_ms=500., max_interval_ms=2000., verbose = False):
    """
    Compute metrics on ecg peaks: HRV_Mean, HRV_SD, HRV_Median, ...
    
    This metrics are a bit more robust that neurokit2 ones because strange interval
    are skiped from the analysis.

    Parameters
    ----------
    ecg_R_peaks: np.array
        Indices of R peaks
    srate: float
        Sampling rate
    min_interval_ms: float (default 500ms)
        Minimum interval inter R peak
    max_interval_ms: float (default 2000ms)
        Maximum interval inter R peak
    verbose: bool (default False)
        Control verbosity
    Returns
    -------
    metrics: pd.Series
        A table contaning metrics
    """
    
    peak_ms = ecg_R_peaks / srate * 1000.
    
    remove = np.zeros(peak_ms.size, dtype='bool')
    d = np.diff(peak_ms) 
    bad, = np.nonzero((d > max_interval_ms)| (d < min_interval_ms))
    remove[bad] = True
    remove[bad+1] = True
    
    peak_ms[remove] = np.nan

    if verbose:
        print(f'{sum(np.isnan(peak_ms))} peaks removed')

    
    delta_ms = np.diff(peak_ms)
    
    # keep = delta_ms < max_interval_ms
    
    # delta_ms = delta_ms[keep]
    
    
    metrics = pd.Series(dtype = float)
    
    metrics['HRV_Mean'] = np.nanmean(delta_ms)
    metrics['HRV_SD'] = np.nanstd(delta_ms)
    metrics['HRV_Median'], metrics['HRV_Mad'] = compute_median_mad(delta_ms[~np.isnan(delta_ms)])
    metrics['HRV_CV'] = metrics['HRV_SD'] / metrics['HRV_Mean']
    metrics['HRV_MCV'] = metrics['HRV_Mad'] / metrics['HRV_Median']
    metrics['HRV_Asymmetry'] = metrics['HRV_Median'] - metrics['HRV_Mean']

    
    # TODO
    metrics['HRV_RMSSD'] = np.sqrt(np.nanmean(np.diff(delta_ms)**2))

    # return pd.DataFrame(metrics).T
    return metrics
    

def compute_instantaneous_rr_interval(ecg_R_peaks, srate, times, min_interval_ms=500., max_interval_ms=2000.,
                                      units='ms', interpolation_kind='linear'):
    """
    Compute the instantaneous RR interval "hrv" signals on a given time vector.
    The output can be interval in units='ms' or frequency in units='bpm'

    Parameters
    ----------
    ecg_R_peaks: np.array
        Indices of R peaks
    srate: float
        Sampling rate
    times: np.array
        The time vector used for interpolation
    max_interval_ms:  float (default 2000.)
        Max RR interval.
    units: 'ms' / 'bpm'
        The units of the interpolated vector.
    interpolation_kind: 'linear' / 'cubic'
        how to interpolate
    Returns
    -------
    hrv: np.array
        The "hrv" signal
    """
    peak_ms = ecg_R_peaks / srate * 1000.

    delta_ms = np.diff(peak_ms)
    keep,  = np.nonzero((delta_ms < max_interval_ms) & (delta_ms > min_interval_ms))

    peak_ms = peak_ms[keep]
    delta_ms = delta_ms[keep]

    peak_s = peak_ms / 1000

    if units == 'ms':
        delta = delta_ms
    elif units == 'bpm':
        delta = 60  / (delta_ms / 1000.)
    else:
        raise ValueError(f'Bad units {units}')


    interp = scipy.interpolate.interp1d(peak_s, delta, kind=interpolation_kind, axis=0,
                                        bounds_error=False, fill_value='extrapolate')
    
    rr_interval = interp(times)

    return rr_interval


def compute_instantaneous_rate(peak_times, new_times, limits=None, units='bpm', interpolation_kind='linear'):
    """
    

    Parameters
    ----------
    peak_times : np.array
        Peak times in seconds
    new_times : np.array
        New vector times
    limits : list or None
        Limits for removing outliers.
    units : 'bpm' / 'Hz' / 'ms' / 's'
        Units of the rate. can be interval or rate.
    interpolation_kind : 'linear'/ 'cubic'

    """
    delta = np.diff(peak_times)

    if units == 's':
        delta = delta
    elif units == 'ms':
        delta = delta * 1000.
    elif units == 'Hz':
        delta = 1.  / delta
    elif units == 'bpm':
        delta = 60.  / delta
    else:
        raise ValueError(f'Bad units {units}')

    if limits is not None:
        lim0, lim1 = limits
        keep,  = np.nonzero((delta > lim0) & (delta < lim1))
        peak_times = peak_times[keep]
        delta = delta[keep]
    else:
        peak_times = peak_times[:-1]


    interp = scipy.interpolate.interp1d(peak_times, delta, kind=interpolation_kind, axis=0,
                                        bounds_error=False, fill_value='extrapolate')
    
    instantaneous_rate = interp(new_times)

    return instantaneous_rate
