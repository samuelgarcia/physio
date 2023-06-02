"""
Some predefined parameters preset for computing respiration and ecg without pain.
"""

import copy

def recursive_update(d, u):
    """
    Recurssive dict update.
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
    preprocess=dict(band=7., btype='lowpass', ftype='bessel', order=5, normalize=False),
    smooth=dict(win_shape='gaussian', sigma_ms=60.0),
    cycle_detection=dict(inspiration_adjust_on_derivative=False),
    baseline=dict(baseline_mode='median'),
    baseline_detect=dict(baseline_mode='median - epsilon'),
    cycle_clean=dict(low_limit_log_ratio=3.5),
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


