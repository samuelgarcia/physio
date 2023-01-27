import numpy as np

def compute_median_mad(data, axis=0):
    """
    Compute median and mad
    """
    med = np.median(data, axis=axis)
    mad = np.median(np.abs(data - med), axis=axis) / 0.6744897501960817
    return med, mad


def detect_peak(traces, srate, thresh=5, exclude_sweep_ms=4.0):
    """
    Simple peak detector.
    """
    
    exclude_sweep_size = int(exclude_sweep_ms / 1000. * srate)
    
    traces_center = traces[exclude_sweep_size:-exclude_sweep_size]
    length = traces_center.shape[0]
    
    med, mad = compute_median_mad(traces)
    abs_threholds = med + thresh * mad
    
    peak_mask = traces_center > abs_threholds
    for i in range(exclude_sweep_size):
        peak_mask &= traces_center > traces[i:i + length]
        peak_mask &= traces_center >= traces[exclude_sweep_size +
                                             i + 1:exclude_sweep_size + i + 1 + length]
    
    inds,  = np.nonzero(peak_mask)
    inds += exclude_sweep_size
    
    return inds

def get_empirical_mode(traces):
    """
    Get the emprical mode of a distribution
    """
    
    q0 = np.quantile(traces, 0.25)
    q1 = np.quantile(traces, 0.75)


    mask = (traces > q0)  & (traces < q1)
    traces = traces[mask]
    count, bins = np.histogram(traces, bins=np.arange(q0, q1, (q1 - q0)/200.))

    ind_max = np.argmax(count)

    mode = bins[ind_max]

    return mode
    


    




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

