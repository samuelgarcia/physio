import numpy as np
import pandas as pd

from .tools import get_empirical_mode, compute_median_mad
from .preprocess import preprocess, smooth_signal
from .parameters import get_respiration_parameters, recursive_update

def compute_respiration(raw_resp, srate, parameter_preset='human_airflow', parameters=None, ):
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
    parameter_preset: str or None
        Name of parameters set 'human_airflow'
        This use the automatic parameters you can also have with get_respiration_parameters('human')
    parameters : dict or None
        When not None this overwrite the parameter set.

    Returns
    -------
    resp: np.array
        A preprocess traces
    cycles: pd.Dataframe
        Table that contain all  cycle information : inspiration/expiration indexes, 
        amplitudes, volumes, durations, ...
    """
    
    if parameter_preset is None:
        params = {}
    else:
        params = get_respiration_parameters(parameter_preset)
    if parameters is not None:
        recursive_update(params, parameters)

    # filter and smooth : more or less 2 times a low pass
    if params['preprocess'] is not None and params['smooth'] is not None:
        center = np.mean(raw_resp)
        resp = raw_resp - center
        if params['preprocess'] is not None:
            resp = preprocess(resp, srate, **params['preprocess'])
        if params['smooth'] is not None:
            resp = smooth_signal(resp, srate, **params['smooth'])
        resp += center
    else:
        resp = raw_resp


    baseline = get_respiration_baseline(resp, srate, **params['baseline'])
    
    
    detection_method = params['cycle_detection']["method"]

    if detection_method == "crossing_baseline":
        cycles = detect_respiration_cycles(resp, srate, 
                                           baseline_mode='manual', baseline=baseline,
                                           **params['cycle_detection'])
    else:
        cycles = detect_respiration_cycles(resp, srate, 
                                            **params['cycle_detection'])


    resp_cycles = compute_respiration_cycle_features(resp, srate, cycles, baseline=baseline, **params["features"])
    
    
    if params['cycle_clean'] is not None:
        resp_cycles = clean_respiration_cycles(resp, srate, resp_cycles, baseline, **params['cycle_clean'])
    
    return resp, resp_cycles


def get_respiration_baseline(resp, srate, baseline_mode='manual', baseline=None):
    """
    Get respiration baseline = respiration mid point.
    This used for:
      * detect_respiration_cycles() for corssing zero
      * compute_respiration_cycle_features() for volume integration
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
    # elif baseline_mode == 'median - epsilon':
    #     epsilon = (np.quantile(resp, 0.75) - np.quantile(resp, 0.25)) / 100.
    #     if baseline is not None:
    #         baseline = baseline - epsilon * 5.
    #     else:
    #         baseline = np.median(resp) - epsilon * 5.
    else:
        raise ValueError(f'get_respiration_baseline wring baseline_mode {baseline_mode}')

    return baseline


def detect_respiration_cycles(resp, srate, method="crossing_baseline", **method_kwargs):
    """
    Detect respiration with several methods:
      * "crossing_baseline": internally use detect_respiration_cycles_crossing_baseline()
      * "co2" : internally use detect_respiration_cycles_co2()

    Parameters
    ----------
    resp: np.array
        Preprocess traces of respiratory signal.
    srate: float
        Sampling rate
    method: 'crossing_baseline' / 'co2'
        Which method for respiration.
    **method_kwargs: 
        All other option are routed to the sub function.
    Returns
    -------
    cycles: np.array
        Indices of inspiration and expiration. shape=(num_cycle, 3)
        with [index_inspi, index_expi, index_next_inspi]
    """    
    if method == "crossing_baseline":
        cycles = detect_respiration_cycles_crossing_baseline(resp, srate, **method_kwargs)
    elif method == "co2":
        cycles = detect_respiration_cycles_co2(resp, srate, **method_kwargs)
    else:
        raise ValueError(f"detect_respiration_cycles(): {method} do not exists")
    
    return cycles


def detect_respiration_cycles_crossing_baseline(resp, srate, baseline_mode='manual', baseline=None, 
                              epsilon_factor1=10, epsilon_factor2=5, inspiration_adjust_on_derivative=False):
    """
    Detect respiration by cycles based on:
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

    baseline = get_respiration_baseline(resp, srate, baseline_mode=baseline_mode, baseline=baseline)

    #~ q90 = np.quantile(resp, 0.90)
    q10 = np.quantile(resp, 0.10)
    epsilon = (baseline - q10) / 100.

    baseline_dw = baseline - epsilon * epsilon_factor1
    baseline_insp = baseline - epsilon * epsilon_factor2

    resp0 = resp[:-1]
    resp1 = resp[1:]

    ind_dw, = np.nonzero((resp0 >= baseline_dw) & (resp1 < baseline_dw))
    
    ind_insp, = np.nonzero((resp0 >= baseline_insp) & (resp1 < baseline_insp))
    ind_insp_no_clean = ind_insp.copy()
    keep_inds = np.searchsorted(ind_insp, ind_dw, side='right')
    keep_inds = keep_inds[keep_inds > 0]
    ind_insp = ind_insp[keep_inds - 1]
    ind_insp = np.unique(ind_insp)
    ind_exp, = np.nonzero((resp0 < baseline) & (resp1 >= baseline))
    
    
    ind_insp, ind_exp = interleave_insp_exp(ind_insp, ind_exp, remove_first_insp=True, remove_first_exp=False)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(resp)
    # ax.scatter(ind_insp, resp[ind_insp], color='g')
    # ax.scatter(ind_exp, resp[ind_exp], color='r')
    # ax.axhline(baseline_dw)
    # ax.axhline(baseline_insp)
    # ax.axhline(baseline)
    # plt.show()
    

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
    
    cycles = np.zeros((ind_insp.size - 1, 3), dtype='int64')
    cycles[:, 0] = ind_insp[:-1]
    cycles[:, 1] = ind_exp
    cycles[:, 2] = ind_insp[1:]


    return cycles


def detect_respiration_cycles_co2(co2_raw, srate, thresh_inspi_factor=0.08, thresh_expi_factor=0.05,
                                  clean_by_mid_value=True):
    """
    Detect respiration for CO2 sensor.

    Parameters
    ----------
    co2_raw: np.array
        Preprocess traces of respiratory signal.
    srate: float
        Sampling rate
    thresh_inspi_factor: float, default 0.05
        Fraction of the min derivative for sertting threshold for inspiration
    thresh_expi_factor: float, default 0.05
        Fraction of the min derivative for sertting threshold for expiration
    clean_by_mid_value: bool, default True
        Remove strange cycle by mid value.

    Returns
    -------
    cycles: np.array
        Indices of inspiration and expiration. shape=(num_cycle, 3)
        with [index_inspi, index_expi, index_next_inspi]
    """ 


    # detect resp on CO2 signal
    co2_gradient = np.gradient(co2_raw)
    min_, max_ = np.min(co2_gradient), np.max(co2_gradient)

    thresh_inspi = min_ * thresh_inspi_factor
    thresh_expi = max_ * thresh_expi_factor
    
    ind_insp = np.flatnonzero((co2_gradient[:-1] >= thresh_inspi) & (co2_gradient[1:] < thresh_inspi))
    ind_exp = np.flatnonzero((co2_gradient[:-1]  <= thresh_expi) & (co2_gradient[1:] > thresh_expi))
    

    ind_insp, ind_exp = interleave_insp_exp(ind_insp, ind_exp, remove_first_insp=True, remove_first_exp=False)
    
    # simple clean by separting up and down values
    if clean_by_mid_value:
        insp_values = co2_raw[ind_insp]
        exp_values = co2_raw[ind_exp]
        mid_value = (np.median(insp_values) + np.median(exp_values)) / 2
        remove_inds = np.flatnonzero(exp_values > mid_value)
        keep = np.ones(ind_insp.size, dtype=bool)
        keep[remove_inds] = False
        ind_insp = ind_insp[keep]
        keep = np.ones(ind_exp.size, dtype=bool)
        keep[remove_inds] = False
        ind_exp = ind_exp[keep]
 
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(nrows=2, sharex=True)
    # ax = axs[0]
    # ax.plot(co2_raw)
    # ax.plot(smoothed_co2)
    # ax.scatter(ind_insp, smoothed_co2[ind_insp], color='g')
    # ax.scatter(ind_exp, smoothed_co2[ind_exp], color='r')
    # ax = axs[1]
    # ax.plot(co2_gradient)
    # ax.axhline(thresh_inspi, color='g')
    # ax.axhline(thresh_expi, color='r')
    # plt.show()

    cycles = np.zeros((ind_insp.size - 1, 3), dtype='int64')
    cycles[:, 0] = ind_insp[:-1]
    cycles[:, 1] = ind_exp
    cycles[:, 2] = ind_insp[1:]
    
    return cycles


def _ensure_interleave(ind0, ind1, remove_first=True):
    """
    Clean ind0 so they are interlevaed into ind1.
    """
    keep_inds = np.searchsorted(ind1, ind0,  side='right')
    keep = np.ones(ind0.size, dtype=bool)
    ind_bad = np.flatnonzero(np.diff(keep_inds) == 0)
    if remove_first:
        keep[ind_bad] = False
    else:
        keep[ind_bad + 1] = False
    ind0_clean = ind0[keep]
    return ind0_clean


def interleave_insp_exp(ind_insp, ind_exp, remove_first_insp=True, remove_first_exp=False):
    """
    Ensure index of inspiration and expiration are interleaved.

    Ensure also that it start and stop with inspiration so that ind_insp.size == ind_exp.size + 1
    """

    ind_exp = _ensure_interleave(ind_exp, ind_insp, remove_first=remove_first_exp)

    ind_insp = _ensure_interleave(ind_insp, ind_exp, remove_first=remove_first_insp)


    if np.any(ind_exp < ind_insp[0]):
        ind_exp = ind_exp[ind_exp>ind_insp[0]]

    if np.any(ind_exp > ind_insp[-1]):
        ind_exp = ind_exp[ind_exp<ind_insp[-1]]

    # corner cases several ind_insp at the beginning/end
    n = np.sum(ind_insp < ind_exp[0])
    if n > 1:
        ind_insp = ind_insp[n - 1:]
    n = np.sum(ind_insp > ind_exp[-1])
    if n > 1:
        ind_insp = ind_insp[: - (n - 1)]
    
    return ind_insp, ind_exp


def compute_respiration_cycle_features(resp, srate, cycles, baseline=None, compute_volume=True, compute_amplitude=True):
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
    resp_cycles: pd.Dataframe
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
    df = resp_cycles = pd.DataFrame(index=index)
    

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
    df['cycle_ratio'] = df['inspi_duration'] / df['cycle_duration']

    if compute_volume:
        for k in ('inspi_volume', 'expi_volume', ):
            df[k] = pd.Series(dtype='float64')

    if compute_amplitude:
        for k in ('total_amplitude', 'inspi_amplitude', 'expi_amplitude'):
            df[k] = pd.Series(dtype='float64')
    
    #missing cycle
    mask = (ix2 == -1)
    df.loc[mask, ['expi_time', 'cycle_duration', 'inspi_duration', 'expi_duration', 'cycle_freq']] = np.nan
    
    if compute_volume or compute_amplitude:
        for c in range(n):
            i1, i2, i3 = ix1[c], ix2[c], ix3[c]
            if i2 == -1:
                #this is a missing cycle in the middle
                continue
            if compute_volume:
                df.at[c, 'inspi_volume'] = np.abs(np.sum(resp[i1:i2])) / srate
                df.at[c, 'expi_volume'] = np.abs(np.sum(resp[i2:i3])) / srate
            if compute_amplitude:
                df.at[c, 'inspi_amplitude'] = np.max(np.abs(resp[i1:i2]))
                df.at[c, 'expi_amplitude'] = np.max(np.abs(resp[i2:i3]))
    
    if compute_amplitude:
        df['total_amplitude'] = df['inspi_amplitude'] + df['expi_amplitude']

    if compute_volume:
        df['total_volume'] = df['inspi_volume'] + df['expi_volume']
    
    return resp_cycles


def clean_respiration_cycles(resp, srate, resp_cycles, baseline, low_limit_log_ratio=3):
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
    resp_cycles: pd.Dataframe
        Features of all cycles given by compute_respiration_cycle_features before clean.
    baseline: 
        The baseline used to recompute resp_cycles

    Returns
    -------
    cleaned_cycles: 
        Clean version of cycles.

    """

    cols = ['inspi_index', 'expi_index', 'next_inspi_index']


    # remove small inspi volumes: remove the current cycle
    log_vol = np.log(resp_cycles['inspi_volume'].values)
    med, mad = compute_median_mad(log_vol)
    limit = med - mad * low_limit_log_ratio
    bad_cycle, = np.nonzero(log_vol < limit)
    keep = np.ones(resp_cycles.shape[0], dtype=bool)
    keep[bad_cycle] = False
    new_cycles = resp_cycles.iloc[keep, :].loc[:, cols].values
    new_cycles[:-1, 2] = new_cycles[1:, 0]

    #~ import matplotlib.pyplot as plt
    #~ fig, axs = plt.subplots(ncols=3)
    #~ ax = axs[0]
    #~ ax.plot(resp)
    #~ inspi_index = resp_cycles['inspi_index'].values
    #~ expi_index = resp_cycles['expi_index'].values
    #~ ax.scatter(inspi_index, resp[inspi_index], marker='o', color='green')
    #~ ax.scatter(expi_index, resp[expi_index], marker='o', color='red')
    #~ ax.scatter(inspi_index[~keep], resp[inspi_index[~keep]], marker='*', color='k', s=500)
    #~ ax = axs[1]
    #~ ax.hist(log_vol, bins=200)
    #~ ax.axvline(limit, color='orange')
    #~ ax.axvspan(med - mad, med + mad, alpha=0.2, color='orange')
    #~ ax = axs[2]
    #~ vol = resp_cycles['inspi_volume'].values
    #~ med2, mad2 = compute_median_mad(vol)
    #~ ax.hist(vol, bins=200)
    #~ ax.axvspan(med2 - mad2, med2 + mad2, alpha=0.1, color='orange')
    #~ plt.show()

    # recompute new volumes and amplitudes
    resp_cycles = compute_respiration_cycle_features(resp, srate, new_cycles, baseline=baseline)
    
    # remove small expi volumes: remove the next cycle
    log_vol = np.log(resp_cycles['expi_volume'].values)
    med, mad = compute_median_mad(log_vol)
    limit = med - mad * low_limit_log_ratio
    bad_cycle, = np.nonzero(log_vol < limit)
    
    # last cycle cannot be removed
    bad_cycle = bad_cycle[bad_cycle < (resp_cycles.shape[0] -1) ]

    # find next good cycle to take expi_index
    for c in bad_cycle:
        next_cycle = c + 1
        while next in bad_cycle:
            next_cycle = c + 1
        #~ if next_cycle < resp_cycles.shape[0]:
        resp_cycles['expi_index'].iat[c] = resp_cycles['expi_index'].iat[next_cycle]
        resp_cycles['next_inspi_index'].iat[c] = resp_cycles['next_inspi_index'].iat[next_cycle]

    bad_cycle += 1
    keep = np.ones(resp_cycles.shape[0], dtype=bool)
    keep[bad_cycle] = False
    new_cycles = resp_cycles.iloc[keep, :].loc[:, cols].values
    new_cycles[:-1, 2] = new_cycles[1:, 0]
    # recompute new volumes and amplitudes
    resp_cycles = compute_respiration_cycle_features(resp, srate, new_cycles, baseline=baseline)

    return resp_cycles
