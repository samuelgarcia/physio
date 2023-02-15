import numpy as np
import scipy.signal

from .tools import compute_median_mad

def preprocess(traces, srate, band=[5., 45.], btype='bandpass', ftype='bessel', order=5, normalize=True):
    """
    Apply simple filter using scipy
    
    For ECG bessel 5-50Hz and order 5 is maybe a good choice.

    By default also normalize the signal using median and mad.


    Parameters
    ----------
    traces: np.array
        Input signal.
    srate: float
        Sampling rate
    band: list of float or float
        Tha band in Hz or scalar if high/low pass.
    btype: 'bandpass', 'highpass', 'lowpass'
        The band pass type
    ftype: str (dfault 'bessel')
        The filter type used to generate coefficient using scipy.signal.iirfilter
    order: int (default 5)
        The order
    normalize: cool (default True)
        Aplly or not normalization
    Returns
    -------
    traces_clean: np.array
        The preprocess traces
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
    """
    A simple signal smoother using gaussian/rect kernel.

    Parameters
    ----------
    traces: np.array
        Input signal.
    srate: float
        Sampling rate
    win_shape: 'gaussian' / 'rect'
        The shape of the kernel
    sigma_ms: float (default 5ms)
        The length of the kernel. Sigma for the gaussian case.
    Returns
    -------
    trace_smooth: np.array
        The smoothed traces
    """

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