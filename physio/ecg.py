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

    Returns
    -------


    """
    clean_ecg = preprocess(raw_ecg, srate, band=[5., 45.], ftype='bessel', order=5, normalize=True)
    
    # TODO estimation du seuil
    
    raw_ecg_peak = detect_peak(clean_ecg, srate, thresh=5, exclude_sweep_ms=4.0)
    
    ecg_peaks = clean_ecg_peak(clean_ecg, srate, raw_ecg_peak)
    
    return clean_ecg, ecg_peaks




def clean_ecg_peak(ecg, srate, raw_peak_inds, min_interval_ms=400.):

    """
    Clean peak with ultra simple idea: remove short interval.


    Parameters
    ----------

    Returns
    -------


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




def compute_ecg_metrics(ecg_peaks, srate, min_interval_ms=500., max_interval_ms=2000., verbose = False):
    """
    Compute metrics on ecg peaks.
    
    This metrics are a bit more robust that neurokit2 ones because strange interval
    are skiped from the analysis.

    Parameters
    ----------

    Returns
    -------


    """
    
    peak_ms = ecg_peaks / srate * 1000.
    
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
    

# compute HRV with resample
def compute_hrv_resampled(ecg_peaks, srate, hrv_times):
    """

    Parameters
    ----------

    Returns
    -------


    """
    peak_times = ecg_peaks / srate

    interp = scipy.interpolate.interp1d(peak_times[:-1], np.diff(peak_times), kind='linear', axis=0,
                                        bounds_error=False, fill_value='extrapolate')
    
    hrv = interp(hrv_times)

    return hrv
