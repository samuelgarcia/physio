"""
Some predefined parameters preset for computing respiration and ecg without pain.
"""

import copy

def recursive_update(d, u):
    """
    Recursive dict update.
    """
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d



def get_respiration_parameters(parameter_preset):
    """
    Get parameters nested dict for a given predefined parameter preset.
    """
    return copy.deepcopy(_resp_parameters[parameter_preset])


def get_ecg_parameters(parameter_preset):
    """
    Get parameters nested dict for a given predefined parameter preset.
    """
    return copy.deepcopy(_ecg_parameters[parameter_preset])


###################################################
# Resp preset

_resp_parameters = {}

# this parameters works with airflow sensor for a human
_resp_parameters['human_airflow'] = dict(
    sensor_type='airflow',
    preprocess=dict(band=7., btype='lowpass', ftype='bessel', order=5, normalize=False),
    smooth=dict(win_shape='gaussian', sigma_ms=60.0),
    cycle_detection=dict(method="crossing_baseline", epsilon_factor1=10., epsilon_factor2=5., inspiration_adjust_on_derivative=False),
    baseline=dict(baseline_mode='median'),
    cycle_clean=dict(variable_names=['inspi_volume', 'expi_volume'], low_limit_log_ratio=4.5),
)
_resp_parameters['human_belt'] = dict(
    sensor_type='belt',
    preprocess=dict(band=5., btype='lowpass', ftype='bessel', order=5, normalize=False),
    smooth=dict(win_shape='gaussian', sigma_ms=40.0),
    cycle_detection=dict(method="min_max", min_cycle_duration_ms=400.),
    baseline=None,
    cycle_clean=dict(variable_names=['inflation_amplitude', 'deflation_amplitude'], low_limit_log_ratio=8.),
)

_resp_parameters['human_co2'] = dict(
    sensor_type='co2',
    preprocess=dict(band=10., btype='lowpass', ftype='bessel', order=5, normalize=False),
    smooth=dict(win_shape='gaussian', sigma_ms=40.0),
    cycle_detection=dict(method="co2", thresh_inspi_factor=0.08, thresh_expi_factor=0.05, clean_by_mid_value=True),
    baseline=dict(baseline_mode='median'),
    cycle_clean=None, # no clean because no volume
)



_resp_parameters['rat_plethysmo'] = dict(
    sensor_type='airflow',
    preprocess=dict(band=30., btype='lowpass', ftype='bessel', order=5, normalize=False),
    smooth=dict(win_shape='gaussian', sigma_ms=5.0),
    #~ smooth=None,
    cycle_detection=dict(method="crossing_baseline", epsilon_factor1=10., epsilon_factor2=4., inspiration_adjust_on_derivative=False),
    baseline=dict(baseline_mode='manual', baseline=0.),
    cycle_clean=dict(variable_names=['inflation_amplitude', 'deflation_amplitude'], low_limit_log_ratio=8.),
)

# belt form etisens https://etisense.com/
_resp_parameters['rat_etisens_belt'] = dict(
    sensor_type='belt',
    preprocess=dict(band=30., btype='lowpass', ftype='bessel', order=5, normalize=False),
    # smooth=dict(win_shape='gaussian', sigma_ms=5.0),
    smooth=None,
    cycle_detection=dict(method="min_max", min_cycle_duration_ms=50.),
    baseline=None,
    cycle_clean=None,
    # this need to be tested
    # cycle_clean=dict(variable_names=["inflation_amplitude", "deflation_amplitude"], low_limit_log_ratio=8.),
)



###################################################
#ECG preset

_ecg_parameters = {}

# this parameters works well with simple ecg signal with positive peaks
_ecg_parameters['human_ecg'] = dict(
    preprocess=dict(band=[5., 45.], ftype='bessel', order=5, normalize=True),
    peak_detection=dict(thresh='auto', exclude_sweep_ms=4.0),
    peak_clean=dict(min_interval_ms=400.),
)

_ecg_parameters['rat_ecg'] = dict(
    preprocess=dict(band=[5., 200.], ftype='bessel', order=5, normalize=True),
    peak_detection=dict(thresh='auto', exclude_sweep_ms=4.0),
    peak_clean=dict(min_interval_ms=50.),
)



