import numpy as np
import pandas as pd

from .tools import get_empirical_mode, compute_median_mad

from .preprocess import preprocess, smooth_signal

def compute_respiration(raw_resp, srate):
    """
    Function for respiration that:
      * preprocess the signal
      * detect cycle
      * clean cycles
      * compute metrics cycle by cycle


    Parameters
    ----------
    raw_resp: np.array
        Raw traces of respiratory signal
    srate: float
        Sampling rate
    Returns
    -------
    resp: np.array
        A preprocess traces
    cycles: pd.Dataframe
        Table that contain all  cycle information : inspiration/expiration indexes, 
        amplitudes, volumes, durations, ...
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


    cycle_features = compute_respiration_cycle_features(resp, srate, cycles, baseline=baseline)

    cycle_features = clean_respiration_cycles(resp, srate, cycle_features, baseline, low_limit_log_ratio=3)
    
    
    return resp, cycle_features



def detect_respiration_cycles(resp, srate, baseline_mode='manual', baseline=None, inspration_ajust_on_derivative=False):
    """
    Detect respiration cycles based on:
      * crossing zeros (or crossing baseline)
      * some cleanning with euristicts

    Parameters
    ----------
    resp: np.array
        Preprocess traces of respiratory signal.
    srate: float
        Sampling rate
    baseline_mode: 'manual' / 'zero' / 'median' / 'mode'
        How to compute the baseline for zero crossings.
    baseline: float or None
        External baseline when baseline_mode='manual'
    inspration_ajust_on_derivative: bool (default False)
        For the inspiration detection, the zero crossing can be refined to auto detect the inflection point.
        This can be usefull when expiration ends with a long plateau.
    Returns
    -------
    cycles: np.array
        Indices of inspiration and expiration. shape=(num_cycle, 3)
        with [index_inspi, index_expi, index_next_inspi]
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
    
    # cycles = np.zeros((ind_insp.size, 2), dtype='int64')
    # cycles[:, 0] = ind_insp
    # cycles[:-1, 1] = ind_exp
    # cycles[-1, 1] = -1

    cycles = np.zeros((ind_insp.size - 1, 3), dtype='int64')
    cycles[:, 0] = ind_insp[:-1]
    cycles[:, 1] = ind_exp
    cycles[:, 2] = ind_insp[1:]


    return cycles






def compute_respiration_cycle_features(resp, srate, cycles, baseline=None):
    """
    Compute respiration features cycle by cycle

    Parameters
    ----------
    resp: np.array
        Preprocess traces of respiratory signal.
    srate: float
        Sampling rate
    cycles: np.array
        Indices of inspiration and expiration. shape=(num_cycle + 1, 2)
    baseline: float or None
        If not None then the baseline is substracted to resp to compute amplitudes and volumes.
    Returns
    -------
    cycle_features: pd.Dataframe
        Features of all cycles.
    """

    if baseline is not None:
        resp = resp - baseline

    times = np.arange(resp.size) / srate

    assert cycles.dtype.kind == 'i'
    assert cycles.ndim == 2
    assert cycles.shape[1] == 3
    
    # n = cycles.shape[0] - 1
    n = cycles.shape[0]
    
    index = np.arange(n, dtype = 'int64')
    df = cycle_features = pd.DataFrame(index=index)
    

    # ix1 = cycles[:-1, 0]
    # ix2 = cycles[:-1, 1]
    # ix3 = cycles[1:, 0]

    ix1 = cycles[:, 0]
    ix2 = cycles[:, 1]
    ix3 = cycles[:, 2]

    # protection
    assert np.all(ix1 < ix2)
    assert np.all(ix2 < ix3)


    t1 = times[ix1]
    t2 = times[ix2]
    t3 = times[ix3]

    # df['start_index'] = pd.Series(ix1 , dtype='int64')
    # df['stop_index'] = pd.Series(ix3 , dtype='int64')
    # df['start_time'] = pd.Series(t1 , dtype='float64')
    # df['stop_time'] = pd.Series(t3 , dtype='float64')

    df['inspi_index'] = pd.Series(ix1 , dtype='int64')
    df['expi_index'] = pd.Series(ix2, dtype='int64')
    df['next_inspi_index'] = pd.Series(ix3 , dtype='int64')
    
    df['inspi_time'] = pd.Series(t1, dtype='float64')
    df['expi_time'] = pd.Series(t2, dtype='float64')
    df['next_inspi_time'] = pd.Series(t3, dtype='float64')

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

        df.at[c, 'inspi_volume'] = np.abs(np.sum(resp[i1:i2])) / srate
        df.at[c, 'expi_volume'] = np.abs(np.sum(resp[i2:i3])) / srate
        df.at[c, 'inspi_amplitude'] = np.max(np.abs(resp[i1:i2]))
        df.at[c, 'expi_amplitude'] = np.max(np.abs(resp[i2:i3]))
    
    df['total_amplitude'] = df['inspi_amplitude'] + df['expi_amplitude']
    
    return cycle_features


def clean_respiration_cycles(resp, srate, cycle_features, baseline, low_limit_log_ratio=3):
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
    resp: np.array
        Preprocess traces of respiratory signal.
    srate: float
        Sampling rate
    cycle_features: pd.Dataframe
        Features of all cycles given by compute_respiration_cycle_features before clean.
    baseline: 
        The baseline used to recompute cycle_features
    Returns
    -------
    cleaned_cycles: 
        Clean version of cycles.
    """

    cols = ['inspi_index', 'expi_index', 'next_inspi_index']


    # remove small inspi volumes: remove the current cycle
    log_vol = np.log(cycle_features['inspi_volume'].values)
    med, mad = compute_median_mad(log_vol)
    limit = med - mad * low_limit_log_ratio
    bad_cycle, = np.nonzero(log_vol < limit)
    keep = np.ones(cycle_features.shape[0], dtype=bool)
    keep[bad_cycle] = False
    new_cycles = cycle_features.iloc[keep, :].loc[:, cols].values
    new_cycles[:-1, 2] = new_cycles[1:, 0]
    # recompute new volumes and amplitudes
    cycle_features = compute_respiration_cycle_features(resp, srate, new_cycles, baseline=baseline)
    
    # remove small expi volumes: remove the next cycle
    log_vol = np.log(cycle_features['expi_volume'].values)
    med, mad = compute_median_mad(log_vol)
    limit = med - mad * low_limit_log_ratio
    bad_cycle, = np.nonzero(log_vol < limit)
    
    # last cycle cannot be removed
    bad_cycle = bad_cycle[bad_cycle < (cycle_features.shape[0] -1) ]

    # find next good cycle to take expi_index
    for c in bad_cycle:
        next_cycle = c + 1
        while next in bad_cycle:
            next_cycle = c + 1
        #~ if next_cycle < cycle_features.shape[0]:
        cycle_features['expi_index'].iat[c] = cycle_features['expi_index'].iat[next_cycle]
        cycle_features['next_inspi_index'].iat[c] = cycle_features['next_inspi_index'].iat[next_cycle]

    bad_cycle += 1
    keep = np.ones(cycle_features.shape[0], dtype=bool)
    keep[bad_cycle] = False
    new_cycles = cycle_features.iloc[keep, :].loc[:, cols].values
    new_cycles[:-1, 2] = new_cycles[1:, 0]
    # recompute new volumes and amplitudes
    cycle_features = compute_respiration_cycle_features(resp, srate, new_cycles, baseline=baseline)

    return cycle_features