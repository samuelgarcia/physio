import numpy as np
import scipy.signal

from .tools import compute_median_mad

def preprocess(traces, sr, band=[5., 45.], ftype='bessel', order=5, normalize=True):
    """
    Apply simple filter using scipy
    
    For ECG bessel 5-50Hz and order 5 is maybe a good choice.
    """
    
    Wn = [e / sr * 2 for e in band]
    filter_coeff = scipy.signal.iirfilter(order, Wn, analog=False, btype='bandpass', ftype=ftype, output='sos')
    
    traces_clean = scipy.signal.sosfiltfilt(filter_coeff, traces, axis=0)
    if normalize:
        med, mad = compute_median_mad(traces_clean)
        traces_clean -= med
        traces_clean /= mad

    return traces_clean 
