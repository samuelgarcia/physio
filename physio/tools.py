import numpy as np

def compute_median_mad(data, axis=0):
    """
    Compute median and mad

    Parameters
    ----------
    data: np.array
        An array
    axis: int (default 0)
        The axis 
    Returns
    -------
    med: 
        The median
    mad: 
        The mad
    """
    med = np.median(data, axis=axis)
    mad = np.median(np.abs(data - med), axis=axis) / 0.6744897501960817
    return med, mad


def detect_peak(traces, srate, thresh=5, abs_threhold=None, exclude_sweep_ms=4.0, thresh_artifact=None, abs_thresh_artifact=None):
    """
    Simple positive peak detector.

    Parameters
    ----------
    traces: np.array
        An array
    srate: float
        Sampling rate of the traces
    thresh: float (default 5)
        The threhold as mad factor
        abs_threhold = med + thresh * mad
    abs_threhold : None or float
        This replace thresh (which is relative to mad)
    exclude_sweep_ms: float
        Zone to exclude multiple peak detection when noisy.
        If several peaks or detected in the same sweep the best is the winner.
    thresh_artifact: float, default None
        ceil threhold that peaks must not exced, this is to remove big artifact
        This is relative to mad
    abs_thresh_artifact: float, None
        Same as abs_thresh_artifact but absolut
    Returns
    -------
    inds: np.array
        Indices on the peaks
    """
    
    exclude_sweep_size = int(exclude_sweep_ms / 1000. * srate)
    exclude_sweep_size = max(exclude_sweep_size, 1)
    
    traces_center = traces[exclude_sweep_size:-exclude_sweep_size]
    length = traces_center.shape[0]
    
    if abs_threhold is None:
        med, mad = compute_median_mad(traces)
        abs_threhold = med + thresh * mad
    
    if abs_thresh_artifact is None and thresh_artifact is not None:
        abs_thresh_artifact = med + thresh_artifact * mad
    
    peak_mask = traces_center > abs_threhold
    if abs_thresh_artifact is not None:
        peak_mask &= (traces_center < abs_thresh_artifact)
        
    for i in range(exclude_sweep_size):
        peak_mask &= traces_center > traces[i:i + length]
        peak_mask &= traces_center >= traces[exclude_sweep_size +
                                             i + 1:exclude_sweep_size + i + 1 + length]
    
    inds,  = np.nonzero(peak_mask)
    inds += exclude_sweep_size
    
    return inds

def get_empirical_mode(traces, nbins=200):
    """
    Get the emprical mode of a distribution.
    This is a really lazy implementation that make an histogram
    of the traces inside quantile [0.25, 0.75] and make an histogram
    of 200 bins and take the max.

    Parameters
    ----------
    traces: np.array
        The traces
    nbins: int (default 200)
        Number of bins for the histogram
    Returns
    -------
    mode: float
        The empirical mode.
    """
    
    q0 = np.quantile(traces, 0.25)
    q1 = np.quantile(traces, 0.75)


    mask = (traces > q0)  & (traces < q1)
    traces = traces[mask]
    count, bins = np.histogram(traces, bins=np.arange(q0, q1, (q1 - q0) / nbins))

    ind_max = np.argmax(count)

    mode = bins[ind_max]

    return mode


def crosscorrelogram(a, b, bins):
    """
    Lazy implementation of crosscorrelogram.
    """
    diff = a[:, np.newaxis] - b[np.newaxis, :]
    count, bins = np.histogram(diff, bins)
    return count, bins




# convoultion suff to keep in mind
# sweep = np.arange(-60, 60)
# wfs = ecg_clean[some_peaks[:, None] + sweep]
# wfs.shape
# kernel = m / np.sum(np.sqrt(m**2))
# kernel -= np.min(kernel)

# kernel -= np.mean(kernel)

# fig, ax = plt.subplots()
# ax.plot(kernel)

# kernel_cmpl = kernel * + kernel * 1j

# ecg_conv = scipy.signal.fftconvolve(ecg_clean, kernel, mode='same')
# ecg_conv = np.convolve(ecg_clean, kernel, mode='same')
# ecg_conv = np.convolve(ecg_clean, kernel_cmpl, mode='same')
# print(ecg_conv.shape, ecg_conv.dtype)
# ecg_conv = np.abs(ecg_conv)

