import numpy as np
import pandas as pd

from .tools import get_empirical_mode

from .preprocess import preprocess, smooth_signal

def compute_respiration(raw_resp, srate, t_start=0.):
    """
    Function for respiration that:
      * preprocess the signal
      * detect cycle
      * clean cycles
      * compute metrics cycle by cycle


    Parameters
    ----------
    raw_resp

    srate

    Returns
    -------

    resp
    
    cycles

    """

    # filter and smooth : more or less 2 times a low pass
    center = np.mean(raw_resp)
    resp = raw_resp - center
    resp = preprocess(resp, srate, band=25., btype='lowpass', ftype='bessel', order=5, normalize=False)
    resp = smooth_signal(resp, srate, win_shape='gaussian', sigma_ms=60.0)
    resp += center
    
    baseline = np.median(resp)

    espilon = (np.quantile(resp, 0.75) - np.quantile(resp, 0.25)) / 100.
    baseline_detect = baseline - espilon * 5.


    cycles = detect_respiration_cycles(resp, srate, baseline_mode='manual', baseline=baseline_detect,
                                       inspration_ajust_on_derivative=False)
    
    cycles = clean_respiration_cycles(resp, srate, cycles)

    cycle_features = compute_respiration_cycle_features(resp, srate, cycles, baseline=baseline, t_start=0.)
    
    
    return resp, cycle_features



def detect_respiration_cycles(resp, srate, baseline_mode='manual', baseline=None,  inspration_ajust_on_derivative=False):
    """
    Detect respiration cycles based on:
      * crossing zeros (or crossing baseline)
      * some cleanning with euristicts

    Parameters
    ----------

    Returns
    -------

    
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

    if inspration_ajust_on_derivative:
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


def clean_respiration_cycles(resp, srate, cycles):
    """
    Remove outlier cycles.
    This is done : 
      * on cycle duration
      * on resp/insp amplitudes
    This can be done with:
      * hard threshold
      * median + K * mad



    Parameters
    ----------

    Returns
    -------

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
    
    # import matplotlib.pyplot as plt
    # count, bins = np.histogram(insp_amplitudes, bins=100)
    # fig, ax = plt.subplots()
    # ax.plot(bins[:-1], count)

    # count, bins = np.histogram(exp_amplitudes, bins=100)
    # fig, ax = plt.subplots()
    # ax.plot(bins[:-1], count)
    
    
    # plt.show()
    
    
    return cleaned_cycles



def compute_respiration_cycle_features(resp, srate, cycles, baseline=None, t_start=0.):
    """
    Compute respiration features cycle by cycle

    Parameters
    ----------
    resp

    srate

    cycles

    baseline

    t_start

    Returns
    -------
    cycle_features: pd.Dataframe
        Features of all cycles.
    """

    if baseline is not None:
        resp = resp - baseline

    times = np.arange(resp.size) / srate + t_start

    assert cycles.dtype.kind == 'i'
    assert cycles.ndim == 2
    assert cycles.shape[1] == 2
    
    n = cycles.shape[0] - 1
    
    index = np.arange(n, dtype = 'int64')
    df = cycle_features = pd.DataFrame(index=index)
    

    ix1 = cycles[:-1, 0]
    ix2 = cycles[:-1, 1]
    ix3 = cycles[1:, 0]

    t1 = times[ix1]
    t2 = times[ix2]
    t3 = times[ix3]

    df['start_index'] = pd.Series(ix1 , dtype='int64')
    df['stop_index'] = pd.Series(ix3 , dtype='int64')
    df['start_time'] = pd.Series(t1 , dtype='float64')
    df['stop_time'] = pd.Series(t3 , dtype='float64')

    df['inspi_index'] = pd.Series(ix1 , dtype='int64')
    df['expi_index'] = pd.Series(ix2, dtype='int64')
    df['inspi_time'] = pd.Series(t1, dtype='float64')
    df['expi_time'] = pd.Series(t2, dtype='float64')
    df['cycle_duration'] = pd.Series(t3 - t1, dtype='float64')
    df['inspi_duration'] = pd.Series(t2 - t1, dtype='float64')
    df['expi_duration'] = pd.Series(t3- t2, dtype='float64')
    df['cycle_freq'] = 1. / df['cycle_duration']
    for k in ('inspi_volume', 'expi_volume', 'total_amplitude', 'inspi_amplitude', 'expi_amplitude'):
        df[k] = pd.Series(dtype='float64')
    
    #missing cycle
    mask = (ix2 == -1)
    df.loc[mask, ['expi_time', 'cycle_duration', 'inspi_duration', 'expi_duration', 'cycle_freq']] = np.nan
    
    for c in range(n):
        i1, i2, i3 = ix1[c], ix2[c], ix3[c]
        if i2 == -1:
            #this is a missing cycle in the middle
            continue
        
        df.at[c, 'insp_volume'] = np.sum(resp[i1:i2]) / srate
        df.at[c, 'exp_volume'] = np.sum(resp[i2:i3]) / srate
        df.at[c, 'insp_amplitude'] = np.max(np.abs(resp[i1:i2]))
        df.at[c, 'exp_amplitude'] = np.max(np.abs(resp[i2:i3]))
    
    df['total_amplitude'] = df['insp_amplitude'] + df['exp_amplitude']
    
    return cycle_features
