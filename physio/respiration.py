import numpy as np
import pandas as pd

from .tools import get_empirical_mode, compute_median_mad, detect_peak
from .preprocess import preprocess, smooth_signal
from .parameters import get_respiration_parameters, recursive_update, possible_resp_preset_txt

import warnings


_possible_sensor_type = ('airflow', 'co2', 'belt') 

def compute_respiration(raw_resp, srate, parameter_preset=None, parameters=None, ):
    """
    Function for respiration that:
      * preprocess the signal
      * detect cycles
      * clean cycles
      * compute metrics cycle by cycle
    
    This function works with 3 types of possible sensors : airflow, belt and co2.
    Depending on these parameters, 3 differents algo will be internatlly used.
    So the `parameters` dict must contain `sensor_type`
    Note that the `resp_cycles`dataframe will contains cycles boundaries and features.
    Features will depend on sensor type.

    See :ref:`handling_parameters` for parameters details.

    Parameters
    ----------

    raw_resp: np.array
        Raw respiratory signal
    srate: float
        Sampling rate of the raw respiratory signal
    parameter_preset: str or None
        Possible presets are : {}
        This string specifies the type of respiratory data, which determines the set of parameters used for processing.
        This set equivalent of the one you get using physio.get_respiration_parameters(preset) with preset being the same one.
    parameters : dict or None
        When not None this updates the parameter set.

    Returns
    -------

    resp: np.array
        A preprocess respiratory trace
    resp_cycles: pd.Dataframe
        resp_cycles is a dataframe containing one row per respiratory cycle and one column per feature (timings, durations, amplitudes, volumnes ...).
    """
    if parameter_preset is None and parameters is None:
        raise ValueError("compute_respiration(): you must give either parameter_preset or parameters (or both!)")

    if parameter_preset is None:
        params = {}
    else:
        params = get_respiration_parameters(parameter_preset)
    
    if parameters is not None:
        recursive_update(params, parameters)
    
    # keep backward compatibility if sensor_type is not provided : 'airflow'
    sensor_type = params.get('sensor_type', 'airflow')
    
    if sensor_type not in _possible_sensor_type:
        raise ValueError(f"sensor_type {sensor_type} is not handled shoudl in : {_possible_sensor_type}")
    

    # filter and smooth : more or less 2 times a low pass
    if params['preprocess'] is not None or params['smooth'] is not None:
        center = np.mean(raw_resp)
        resp = raw_resp - center
        if params['preprocess'] is not None:
            resp = preprocess(resp, srate, **params['preprocess'])
        if params['smooth'] is not None:
            resp = smooth_signal(resp, srate, **params['smooth'])
        resp += center
    else:
        resp = raw_resp


    

    detection_method = params['cycle_detection'].get("method", None)
    if detection_method is None:
        if sensor_type == 'airflow':
            detection_method = 'crossing_baseline'
            
        elif sensor_type == 'belt':
            detection_method = 'min_max'
            baseline = None
        elif sensor_type == 'co2':
            detection_method = 'co2'
            baseline = None
        params['cycle_detection']['detection_method'] = detection_method
    
    if detection_method == 'crossing_baseline':
        baseline = get_respiration_baseline(resp, srate, **params['baseline'])
    else:
        baseline = None


    if detection_method == "crossing_baseline":
        cycles = detect_respiration_cycles(resp, srate, 
                                           baseline_mode='manual', baseline=baseline,
                                           **params['cycle_detection'])
    else:
        cycles = detect_respiration_cycles(resp, srate, 
                                            **params['cycle_detection'])

    resp_cycles = compute_respiration_cycle_features(resp, srate, cycles, baseline=baseline, sensor_type=sensor_type)

    
    if params.get('cycle_clean', None) is not None:
        if isinstance(params['cycle_clean'], dict):
            resp_cycles = clean_respiration_cycles(resp, srate, resp_cycles, baseline=baseline, sensor_type=sensor_type, **params['cycle_clean'])
        elif isinstance(params['cycle_clean'], list):
            
            for p in params['cycle_clean']:
                resp_cycles = clean_respiration_cycles(resp, srate, resp_cycles, baseline=baseline, sensor_type=sensor_type, **p)
        else:
            raise ValueError('params cycle_clean is wrong')
    
    return resp, resp_cycles



compute_respiration.__doc__ = compute_respiration.__doc__.format(possible_resp_preset_txt)


def get_respiration_baseline(resp, srate, baseline_mode='manual', baseline=None):
    """
    Get respiration baseline = respiration mid point.

    This is used for:

      * detect_respiration_cycles() for crossing zero
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
    
    baseline: float
        Value of the computed baseline level
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
    Detect respiration with several possible methods:
      * "crossing_baseline": method used when the respiratory signal is airflow
        internally uses detect_respiration_cycles_crossing_baseline()
      * "min_max" : method used when the respiratory signal is volume
        internally uses detect_respiration_cycles_min_max()
      * "co2" : method used when the respiratory signal is from co2 sensor
        internally uses detect_respiration_cycles_co2()
      

    Parameters
    ----------

    resp: np.array
        Preprocessed respiratory signal.
    srate: float
        Sampling rate
    method: 'crossing_baseline' | 'co2' | 'min_max'
        Which method is used for respiratory cycle detection respiration.
    **method_kwargs: 
        All other options are routed to the sub-function.

    Returns
    -------
    
    cycles: np.array
        Indices of inspiration and expiration. shape=(num_cycle, 3)
        with [index_inspi, index_expi, index_next_inspi]
    """    
    if method == "crossing_baseline":
        cycles = detect_respiration_cycles_crossing_baseline(resp, srate, **method_kwargs)
    elif method == "min_max":
        cycles = detect_respiration_cycles_min_max(resp, srate, **method_kwargs)
    elif method == "co2":
        cycles = detect_respiration_cycles_co2(resp, srate, **method_kwargs)    
    else:
        raise ValueError(f"detect_respiration_cycles(): {method} do not exists")
    
    return cycles


def detect_respiration_cycles_crossing_baseline(resp, srate, baseline_mode='manual', baseline=None, 
                              epsilon_factor1=10., epsilon_factor2=5., inspiration_adjust_on_derivative=False):
    """
    Detect respiration by cycles based on:
      * crossing zeros (or crossing baseline)
      * some cleaning with euristicts

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
    epsilon_factor1: float, default 10.
        Defines a horizontal confidence zone just below the true baseline, where the low part = 
        baseline - `epsilon` * `epsilon_factor1`, with `epsilon` = (baseline - np.quantile(resp, 0.1)) / 100.
    epsilon_factor2: float, default 5.
        Defines the higher part of the confidence zone: baseline - `epsilon` * `epsilon_factor2`. `epsilon_factor1` 
        is higher than `epsilon_factor2` to search the upper part of the confidence zone.
    inspiration_adjust_on_derivative: bool, default False
        For the inspiration detection, the zero crossing can be refined to auto-detect the inflection point.
        This can be useful when expiration ends with a long and shortly drifting plateau.

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
    # ind_insp_no_clean = ind_insp.copy()
    keep_inds = np.searchsorted(ind_insp, ind_dw, side='right')
    keep_inds = keep_inds[keep_inds > 0]
    ind_insp = ind_insp[keep_inds - 1]
    ind_insp = np.unique(ind_insp)
    # ind_insp_no_clean2 = ind_insp.copy()
    ind_exp, = np.nonzero((resp0 < baseline) & (resp1 >= baseline))
    
    
    # ind_insp, ind_exp = interleave_insp_exp(ind_insp, ind_exp, remove_first_insp=True, remove_first_exp=False)
    ind_insp, ind_exp = interleave_insp_exp(ind_insp, ind_exp, remove_first_insp=False, remove_first_exp=False)

    # import matplotlib.pyplot as plt
    # myparams = {
    #     'axes.titlesize' : 14,
    #     'axes.labelsize' : 14,
    #     'xtick.labelsize' : 14,
    #     'ytick.labelsize' : 14
    # }
    # import matplotlib as mpl
    # with mpl.rc_context(myparams):
    #     fig, ax = plt.subplots()
    #     ax.plot(resp, color='black', lw=3)
    #     ax.axhline(baseline_dw, color='C2', label='epsilon 1', lw=3)
    #     ax.axhline(baseline_insp, color='C1', label='epsilon 2', lw=3)
    #     ax.axhline(baseline, color='C0', label='baseline', lw=3)
    #     # ax.axhspan(baseline, baseline_insp, color='C1', alpha=0.3 )
    #     ax.axhspan(baseline, baseline_dw, color='C2', alpha=0.2 )
    #     ax.scatter(ind_exp, resp[ind_exp], color='r', s=120, zorder=100)
    #     ax.scatter(ind_dw, resp[ind_dw], color="#00BD1F", marker='^', s=120, zorder=100)
    #     # ax.scatter(ind_insp_no_clean, resp[ind_insp_no_clean], color='k', marker='^')
    #     ax.scatter(ind_insp_no_clean2, resp[ind_insp_no_clean2], color='m', marker='*', s=120, zorder=100)
    #     ax.scatter(ind_insp, resp[ind_insp], color='g', s=120, zorder=100)
    #     ax.set_xticks([])
    #     ax.set_yticks([])

    #     ax.legend(fontsize=14)
    #     plt.show()
    

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


def detect_respiration_cycles_min_max(resp, srate, exclude_sweep_ms=50.):
    """
    Detect respiration by cycles based on:
      * crossing zeros (or crossing baseline)
      * some cleaning with euristicts

    Parameters
    ----------

    resp: np.array
        Preprocess traces of respiratory signal.
    srate: float
        Sampling rate
    exclude_sweep_ms: float
        Duration in milliseconds of a window sept on the signal to remove too narrow peaks (transition points) in term of horizontal distance.

    Returns
    -------

    cycles: np.array
        Indices of inspiration and expiration. shape=(num_cycle, 3)
        with [index_inspi, index_expi, index_next_inspi]
    """

    # abs_threshold = 0
    abs_threshold = np.min(resp)
    

    ind_exp = detect_peak(resp, srate, abs_threshold=abs_threshold, exclude_sweep_ms=exclude_sweep_ms)
    ind_insp = np.zeros(ind_exp.size - 1, dtype="int64")
    for i in range(ind_exp.size - 1):
        ind_insp[i] = np.argmin(resp[ind_exp[i]:ind_exp[i+1]]) + ind_exp[i]

    cycles = np.zeros((ind_insp.size - 1, 3), dtype='int64')
    cycles[:, 0] = ind_insp[:-1]
    cycles[:, 1] = ind_exp[1:-1]
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
        Fraction of the min derivative for setting threshold for inspiration
    thresh_expi_factor: float, default 0.05
        Fraction of the min derivative for setting threshold for expiration
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
    # ax.set_title('CO2 signal')
    # ax.plot(co2_raw)
    # ax.scatter(ind_insp, co2_raw[ind_insp], color='g')
    # ax.scatter(ind_exp, co2_raw[ind_exp], color='r')
    # ax = axs[1]
    # ax.set_title('CO2 derivative signal')
    # ax.plot(co2_gradient)
    # ax.axhline(thresh_inspi, color='g', label=f"thresh_inspi_factor={thresh_inspi_factor}")
    # ax.axhline(thresh_expi, color='r', label=f"thresh_expi_factor={thresh_expi_factor}")
    # ax.legend(fontsize=14)
    # plt.show()

    cycles = np.zeros((ind_insp.size - 1, 3), dtype='int64')
    cycles[:, 0] = ind_insp[:-1]
    cycles[:, 1] = ind_exp
    cycles[:, 2] = ind_insp[1:]
    
    return cycles


def _ensure_interleave(ind0, ind1, remove_first=True):
    """
    Clean ind0 so they are interleaved into ind1.
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


def compute_respiration_cycle_features(resp, srate, cycles, baseline=None, sensor_type='airflow'):
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
        If not None then the baseline is subtracted to resp to compute amplitudes and volumes.
    sensor_type: str
        The sensor type. Can be one of : 'airflow' | 'belt' | 'co2' 

    Returns
    -------

    resp_cycles: pd.Dataframe
        Features of all cycles.
    """


    if sensor_type == 'airflow':
        compute_volume = True
        compute_amplitude = True
        compute_belt_amplitude = False
    elif sensor_type == 'belt':
        compute_volume = False
        compute_amplitude = False
        compute_belt_amplitude = True
    elif sensor_type == 'co2':
        compute_volume = False
        compute_amplitude = False
        compute_belt_amplitude = False
    else:
        raise ValueError("compute_respiration_cycle_features need sensor_type")


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
        for k in ('inspi_peak_index', 'expi_peak_index'):
            df[k] = pd.Series(np.zeros(n, dtype='int64'), dtype='int64')
        for k in ('inspi_peak_time', 'expi_peak_time'):
            df[k] = pd.Series(dtype='float64')


    if compute_belt_amplitude:
        for k in ('inspi_amplitude', 'expi_amplitude'):
            df[k] = pd.Series(dtype='float64')

    #missing cycle
    mask = (ix2 == -1)
    df.loc[mask, ['expi_time', 'cycle_duration', 'inspi_duration', 'expi_duration', 'cycle_freq']] = np.nan
    
    if compute_volume or compute_amplitude or compute_belt_amplitude:
        for c in range(n):
            i1, i2, i3 = ix1[c], ix2[c], ix3[c]
            if i2 == -1:
                #this is a missing cycle in the middle
                continue
            if compute_volume:
                mask = resp[i1:i2] < 0.
                df.at[c, 'inspi_volume'] = np.abs(np.sum(resp[i1:i2][mask])) / srate
                mask = resp[i2:i3] > 0.
                df.at[c, 'expi_volume'] = np.abs(np.sum(resp[i2:i3][mask])) / srate
            if compute_amplitude:
                ind_max = np.argmax(np.abs(resp[i1:i2]))
                df.at[c, 'inspi_amplitude'] = np.abs(resp[i1+ind_max])
                df.at[c, 'inspi_peak_index'] = i1 + ind_max

                ind_max = np.argmax(np.abs(resp[i2:i3]))
                df.at[c, 'expi_amplitude'] = np.abs(resp[i2+ind_max])
                df.at[c, 'expi_peak_index'] = i2 + ind_max

            if compute_belt_amplitude:
                df.at[c, 'inspi_amplitude'] = resp[i2] - resp[i1]
                df.at[c, 'expi_amplitude'] = resp[i2] - resp[i3]
    
    if compute_amplitude:
        df['total_amplitude'] = df['inspi_amplitude'] + df['expi_amplitude']

    if compute_volume:
        df['total_volume'] = df['inspi_volume'] + df['expi_volume']

    if 'inspi_peak_index' in df.columns:
        df.loc[:, 'inspi_peak_time'] = times[df.loc[:, 'inspi_peak_index'].values]
        df.loc[:, 'expi_peak_time'] = times[df.loc[:, 'expi_peak_index'].values]
    

    return resp_cycles


def clean_respiration_cycles(resp, srate, resp_cycles, baseline=None, variable_names=None, low_limit_log_ratio=3., sensor_type=None):
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
        The baseline used to recompute resp_cycles, this is needed when sensor_type="airflow"
    variable_names: list of str
        Which columns used to search for small deviant
    low_limit_log_ratio: float
        Used to compute low limit with "limit = med - mad * low_limit_log_ratio"
    sensor_type: 'airflow' | 'belt' | 'co2'
        sensor type

    Returns
    -------

    cleaned_cycles: 
        Clean version of cycles.
    """


    if variable_names is None:
        warnings.warn("clean_respiration_cycles() need variable_names to be set, variable_name=['inspi_volume', 'expi_volume'] is set to  for backward compatibility")
        variable_names = ['inspi_volume', 'expi_volume']

    assert isinstance(variable_names, list), "variable_names must be a list of columns"

    index_cols = ['inspi_index', 'expi_index', 'next_inspi_index']

    for variable_name in variable_names:
        log_values = np.log(resp_cycles.loc[:, variable_name].values)
        med, mad = compute_median_mad(log_values)
        limit = med - mad * low_limit_log_ratio
        bad_cycle, = np.nonzero(log_values < limit)


        keep = np.ones(resp_cycles.shape[0], dtype=bool)
        new_cycles = resp_cycles.loc[:, index_cols].values

        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(ncols=3)
        # ax = axs[0]
        # ax.plot(resp)
        # inspi_index = resp_cycles['inspi_index'].values
        # expi_index = resp_cycles['expi_index'].values
        # ax.scatter(inspi_index, resp[inspi_index], marker='o', color='green')
        # ax.scatter(expi_index, resp[expi_index], marker='o', color='red')
        # keep2 = keep.copy()
        # keep2[bad_cycle] = False
        # ax.scatter(inspi_index[~keep2], resp[inspi_index[~keep2]], marker='*', color='k', s=500)
        # ax = axs[1]
        # ax.set_title(f'log {variable_name}')
        # ax.hist(log_values, bins=200)
        # ax.axvline(limit, color='orange')
        # ax.axvspan(med - mad, med + mad, alpha=0.2, color='orange')
        # ax = axs[2]
        # ax.set_title(f'{variable_name} {bad_cycle.size}')
        # vol = resp_cycles[variable_name].values
        # med2, mad2 = compute_median_mad(vol)
        # ax.hist(vol, bins=200)
        # ax.axvspan(med2 - mad2, med2 + mad2, alpha=0.1, color='orange')
        # ax.axvline(np.exp(limit), color='orange')
        # plt.show()


        for c in bad_cycle:
            if not keep[c]:
                # already remove
                continue

            prev_cycle = c - 1
            while prev_cycle in bad_cycle:
                prev_cycle -= 1
            
            prev_cycle = max(prev_cycle, 0)
            
            next_cycle = c + 1
            while next_cycle in bad_cycle:
                next_cycle += 1
            next_cycle = min(next_cycle, resp_cycles.shape[0] -1 )

            if sensor_type == 'airflow':
                # cycle remove
                if 'inspi' in variable_name:
                    new_cycles[prev_cycle, 2] = new_cycles[next_cycle, 0]
                elif 'expi' in variable_name:
                    new_cycles[prev_cycle, 2] = new_cycles[next_cycle, 0]
                    # new_cycles[next_cycle, 0] = new_cycles[prev_cycle, 2]
                else:
                    raise ValueError(f'clean_respiration_cycle do not support variable_name={variable_name}')

                # new_cycles[prev_cycle, 1] = new_cycles[c, 1]
                
            elif sensor_type == 'belt':
                # find minima and move inspi_index
                possible_inds = np.arange(prev_cycle+1, next_cycle+1)
                best = np.argmin(resp[new_cycles[possible_inds, 0]])
                best_ind = possible_inds[best]
                new_cycles[prev_cycle, 2] = new_cycles[best_ind, 0]
                new_cycles[next_cycle, 0] = new_cycles[best_ind, 0]
                # find maxima and move expi_index for next and prev
                ind = np.argmax(resp[new_cycles[prev_cycle, 0]:new_cycles[prev_cycle, 2]])
                new_cycles[prev_cycle, 1] = new_cycles[prev_cycle, 0] + ind
                ind = np.argmax(resp[new_cycles[next_cycle, 0]:new_cycles[next_cycle, 2]])
                new_cycles[next_cycle, 1] = new_cycles[next_cycle, 0] + ind

            elif sensor_type == 'co2':
                # TODO 
                raise NotImplementedError("Clean resp cycle with CO2 is not implemented yet")
            
            # remove it
            keep[prev_cycle+1:next_cycle] = False

        

        keep[bad_cycle] = False
        new_cycles = new_cycles[keep, :]

        resp_cycles = compute_respiration_cycle_features(resp, srate, new_cycles, baseline=baseline, sensor_type=sensor_type)
            
    return resp_cycles
