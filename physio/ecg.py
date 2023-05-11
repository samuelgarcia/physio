import numpy as np
import pandas as pd

import scipy.interpolate

from .tools import detect_peak, compute_median_mad
from .preprocess import preprocess

import warnings



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
    
    
    metrics = pd.Series(dtype=float)
    
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
    

def compute_hrv_psd(peak_times, ecg_duration_s,  sample_rate=100., limits=None, units='bpm',
                                        freqency_bands = {'lf': (0.04, .15), 'hf' : (0.15, .4)},
                                        window_s=250., interpolation_kind='cubic'):
    """
    Compute hrv power spectrum density and extract some metrics:
      * lf power
      * hf power
    
    Please note:
        1. The duration of the signal and the window are important parameters to estimate low frequencies
           in a spectrum. Some warnings or errors should popup if they are too short.
        2. Given that the hrv is mainly driven by the respiration the frequency boudaries are often innacurate!
           For instance a slow respiration at 0.1Hz is moving out from the 'hf' band wheras this band should capture
           the respiratory part of the hrv.
        3. The instataneous rate is computed by interpolating eccg peak interval, the interpolation method
           'linear' or 'cubic' are a real impact of the dynamic and signal smoothness and so the spectrum should differ
           because of the wieight of the harmonics
        4. The units of the instantaneous hrv (bpm, interval in second, interval in ms) have a high impact on the
           magnitude of metrics. Many toolboxes (neurokit2, ) differ a lot on this important detail.
        5. Here we choose the classical welch method for spectrum density estimation. Some parameters have also small
           impact on the results : dentend, windowing, overlap.
    
    
    Parameters
    ----------
    peak_times
    
    ecg_duration_s
    
    sample_rate=100.
    
    limits=None
    
    units='bpm'

    interpolation_kind
    
    """
    

    # See https://github.com/scipy/scipy/issues/8368 about density vs spectrum
    
    
    times = np.arange(0, ecg_duration_s, 1 / sample_rate)
    
    instantaneous_rate = compute_instantaneous_rate(peak_times, times, limits=limits, units=units,
                                                    interpolation_kind=interpolation_kind)
    
    # some check on the window
    min_freq = min(freqs[0] for freqs in freqency_bands.values())
    if window_s <  (1 / min_freq) * 5:
        raise ValueError(f'The window is too short {window_s}s compared to the lowest frequency {min_freq}Hz')
    if ecg_duration_s <  (1 / min_freq) * 5:
        raise ValueError(f'The duration is too short {ecg_duration_s}s compared to the lowest frequency {min_freq}Hz')

    if window_s <  (1 / min_freq) * 10 or (1 / min_freq) * 10:
        warnings.warn(f'The window is not optimal {window_s}s compared to the lowest frequency {min_freq}Hz')
    if ecg_duration_s <  (1 / min_freq) * 10 or (1 / min_freq) * 10:
        warnings.warn(f'The duration is not optimal {ecg_duration_s}s compared to the lowest frequency {min_freq}Hz')

    
    # important note : when using welch with scaling='density'
    # then the integrale (trapz) must be aware of the dx to take in account
    # so the metrics scale is invariant given against sampling rate and also sample_rate
    nperseg = int(window_s * sample_rate)
    nfft = nperseg
    psd_freqs, psd = scipy.signal.welch(instantaneous_rate, detrend='constant', fs=sample_rate, window='hann',
                                                            scaling='density', nperseg=nperseg, noverlap=0, nfft=nfft)

    metrics = pd.Series(dtype=float)
    delta_freq = np.mean(np.diff(psd_freqs))
    for name, freq_band in freqency_bands.items():
        f0, f1 = freq_band
        area = np.trapz(psd[(psd_freqs >= f0) & (psd_freqs < f1)], dx=delta_freq)
        metrics[name] = area

    return psd_freqs, psd, metrics

