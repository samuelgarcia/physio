import numpy as np
import pandas as pd

from .tools import detect_peak, compute_median_mad
from .preprocess import preprocess



def compute_ecg(raw_ecg, srate):
    """
    Function for ECG that:
      * preprocess the ECG
      * detect R peaks
      * apply some cleaning to remove too small ECG interval
      

    """
    clean_ecg = preprocess(raw_ecg, srate, band=[5., 45.], ftype='bessel', order=5, normalize=True)
    
    # TODO estimation du seuil
    
    raw_ecg_peak = detect_peak(clean_ecg, srate, thresh=5, exclude_sweep_ms=4.0)
    
    ecg_peaks = clean_ecg_peak(clean_ecg, srate, raw_ecg_peak)
    
    return clean_ecg, ecg_peaks




def clean_ecg_peak(ecg, srate, raw_peak_inds, min_interval_ms=400.):

    """
    Clean peak with ultra simple idea: remove short interval.
    
    """
    
    # TODO clean plus malin avec le max des deux peak
    
    peak_ms = (raw_peak_inds / srate * 1000.)
    bad_peak, = np.nonzero(np.diff(peak_ms) < min_interval_ms)
    bad_peak += 1
    
    keep = np.ones(raw_peak_inds.size, dtype='bool')
    keep[bad_peak] = False
    peak_inds = raw_peak_inds[keep]
    
    return peak_inds




def compute_ecg_metrics(ecg_peaks, srate, min_interval_ms=500., max_interval_ms=2000., verbose = False):
    """
    Compute metrics on ecg peaks.
    
    This metrics are a bit more robust that neurokit2 ones because strange interval
    are skiped from the analysis.
    
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


    
    
    return pd.DataFrame(metrics).T

def compute_RSA(fci):
    """
    Compute respiratory sinusal arrythmia from instantaneous cardiac frequency signal

    ----------
    Input =
    - fci : instantaneous cardiac frequency signal, ideally in beats per minute = 1D np vector

    Output =
    - median of peaks - throughs values of respiratory induced variations of the fci signal = float
    """

    derivative = np.gradient(fci) # get derivative of signal

    rises, = np.where((derivative[:-1] <=0) & (derivative[1:] >0)) # detect where sign inversion from - to +
    decays, = np.where((derivative[:-1] >=0) & (derivative[1:] <0)) # detect where sign inversion from + to -

    if rises[0] > decays[0]: # first point detected has to be a rise
        decays = decays[1:] # so remove the first decay if is before first rise
    if rises[-1] > decays[-1]: # last point detected has to be a decay
        rises = rises[:-1] # so remove the last rise if is after last decay

    amplitudes_rsa = fci[decays] - fci[rises]
    return np.median(amplitudes_rsa)

# compute HRV with resample
