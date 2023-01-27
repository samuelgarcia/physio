import numpy as np
import scipy.signal

from .tools import compute_median_mad

def preprocess(traces, srate, band=[5., 45.], btype='bandpass', ftype='bessel', order=5, normalize=True):
    """
    Apply simple filter using scipy
    
    For ECG bessel 5-50Hz and order 5 is maybe a good choice.
    """
    
    if np.isscalar(band):
        Wn = band / srate * 2
    else:
        Wn = [e / srate * 2 for e in band]
    filter_coeff = scipy.signal.iirfilter(order, Wn, analog=False, btype=btype, ftype=ftype, output='sos')
    
    traces_clean = scipy.signal.sosfiltfilt(filter_coeff, traces, axis=0)
    if normalize:
        med, mad = compute_median_mad(traces_clean)
        traces_clean -= med
        traces_clean /= mad

    return traces_clean 


def smooth_signal(trace, srate, win_shape='gaussian', sigma_ms=5.0):

    size = int(srate * sigma_ms / 1000.)
    if win_shape == 'gaussian':
        times = np.arange(- 5 * size, 5 * size + 1)
        kernel = np.exp(- times ** 2 / size ** 2)
        kernel /= np.sum(kernel)

    elif win_shape == 'rect':
        kernel = np.ones(size, dtype='folat64') / size
    else:
        raise ValueError(f'Bad win_shape {win_shape}')

    trace_smooth = scipy.signal.fftconvolve(trace, kernel, mode='same', axes=0)

    return trace_smooth