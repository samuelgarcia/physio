import numpy as np
from .tools import get_empirical_mode
from .preprocess import preprocess, smooth_signal


def compute_respiration(raw_resp, srate, show = False):
    """
    Function for respiration that:
      * preprocess the signal
      * detect cycle
      * clean cycles
      * compute metrics cycle by cycle
    """

    # filter and smooth : more or less 2 times a low pass
    resp = preprocess(raw_resp, srate, band=25., btype='lowpass', ftype='bessel', order=5, normalize=False)
    resp = smooth_signal(resp, srate, win_shape='gaussian', sigma_ms=60.0)
    
    cycles = detect_respiration_cycles(resp, srate, baseline_mode='median', baseline=None,  inspiration_adjust_on_derivative=False)
    
    cycles = clean_respiration_cycles(resp, srate, cycles, show = show)
    
    
    return resp, cycles



def detect_respiration_cycles(resp, srate, baseline_mode='manual', baseline=None,  inspiration_adjust_on_derivative=False):
    """
    Detect respiration cycles based on:
      * crossing zeros (or crossing baseline)
      * some cleanning with euristicts

    
    """

    if baseline_mode == 'manual':
        assert baseline is not None
    elif baseline_mode == 'zero':
        baseline = 0.
    elif baseline_mode == 'median':
        baseline = np.median(resp)
    elif baseline_mode == 'mode':
        baseline = get_empirical_mode(resp)

    resp0 = resp[:-1]
    resp1 = resp[1:]

    ind_insp, = np.nonzero((resp0 >= baseline) & (resp1 < baseline))
    ind_exp, = np.nonzero((resp0 < baseline) & (resp1 >= baseline))

    if ind_insp.size == 0:
        print('no cycle dettected')
        return
    
    mask = (ind_exp > ind_insp[0]) & (ind_exp < ind_insp[-1])
    ind_exp = ind_exp[mask]

    if inspiration_adjust_on_derivative:
        # lets find local minima on second derivative
        # this can be slow
        delta_ms = 10.
        delta = int(delta_ms * srate / 1000.)
        derivate1 = np.gradient(resp)
        derivate2 = np.gradient(derivate1)
        for i in range(ind_exp.size):
            i0, i1 = ind_insp[i], ind_exp[i]
            i0 = max(0, i0 - delta)
            i1 = i0 + np.argmin(resp[i0:i1])
            d1 = derivate1[i0:i1]
            i1 = i0 + np.argmin(d1)
            if (i1 - i0) >2:
                # find the last crossing zeros in this this short segment
                d2 = derivate2[i0:i1]
                i1 = i0 + np.argmin(d2)
                if (i1 - i0) >2:
                    d2 = derivate2[i0:i1]
                    mask = (d2[:-1] >=0) & (d2[1:] < 0)
                    if np.any(mask):
                        ind_insp[i] = i0 + np.nonzero(mask)[0][-1]
                
            
            #~ import matplotlib.pyplot as plt
            #~ fig, axs = plt.subplots(nrows=3, sharex=True)
            #~ i1 = ind_exp[i]
            #~ axs[0].plot(resp[i0: i1])
            #~ l = ind_insp[i] - i0
            #~ axs[0].axvline(l)
            #~ axs[1].plot(derivate1[i0:i1])
            #~ axs[1].plot(d1)
            #~ axs[1].axvline(l)
            #~ axs[2].plot(derivate2[i0:i1])
            #~ axs[2].plot(d2)
            #~ axs[2].axvline(l)
            #~ axs[2].axhline(0)
            #~ plt.show()

    cycles = np.zeros((ind_insp.size, 2), dtype='int64')
    cycles[:, 0] = ind_insp
    cycles[:-1, 1] = ind_exp
    cycles[-1, 1] = -1


    return cycles


def clean_respiration_cycles(resp, srate, cycles, show = False):
    """
    Remove outlier cycles.
    This is done : 
      * on cycle duration
      * on resp/insp amplitudes
    This can be done with:
      * hard threshold
      * median + K * mad
    """
    n = cycles.shape[0] - 1
    insp_amplitudes = np.zeros(n)
    exp_amplitudes = np.zeros(n)
    for i in range(n):
        i0, i1, i2 = cycles[i, 0], cycles[i, 1],cycles[i+1, 0]
        insp_amplitudes[i] = np.abs(np.min(resp[i0:i1]))
        exp_amplitudes[i] = np.abs(np.max(resp[i1:i2]))

    cleaned_cycles = cycles
    delta = np.diff(cycles[:, 0])
    
    if show:
        import matplotlib.pyplot as plt
        count, bins = np.histogram(insp_amplitudes, bins=100)
        fig, ax = plt.subplots()
        ax.plot(bins[:-1], count)

        count, bins = np.histogram(exp_amplitudes, bins=100)
        fig, ax = plt.subplots()
        ax.plot(bins[:-1], count)


        plt.show()

    
    return cleaned_cycles


def compute_resp_features(resp, cycles, srate):
    features = []
    for i in range(cycles.shape[0] - 1):
        start = cycles[i,0]
        transition = cycles[i,1]
        stop = cycles[i+1,0]
        start_t = start / srate
        transition_t = transition / srate
        stop_t = stop / srate
        cycle_duration = stop_t - start_t
        inspi_duration = transition_t - start_t
        expi_duration = stop_t - transition_t
        cycle_freq = 1 / cycle_duration
        cycle_ratio = inspi_duration / cycle_duration
        inspi_amplitude = np.max(np.abs(sig[start:transition]))
        expi_amplitude = np.max(np.abs(sig[transition:stop]))
        cycle_amplitude = inspi_amplitude + expi_amplitude
        inspi_volume = np.trapz(np.abs(sig[start:transition]))
        expi_volume = np.trapz(np.abs(sig[transition:stop]))
        cycle_volume = inspi_volume + expi_volume

        features.append([start, transition , stop, start_t, transition_t, stop_t, cycle_duration,
                            inspi_duration, expi_duration, cycle_freq, cycle_ratio, inspi_amplitude,
                            expi_amplitude,cycle_amplitude, inspi_volume, expi_volume, cycle_volume])

    df_features = pd.DataFrame(features, columns = ['start','transition','stop','start_time','transition_time',
                                                    'stop_time','cycle_duration','inspi_duration','expi_duration','cycle_freq','cycle_ratio',
                                                    'inspi_amplitude','expi_amplitude','cycle_amplitude','inspi_volume','expi_volume','cycle_volume'])
    return df_features

