Handling parameters
=======================

Why such a section ?
--------------------

For ECG or Respiration signal processing, :py:mod:`physio` proposes :py:func:`~physio.compute_ecg` and :py:func:`~physio.compute_respiration`,
some high-level wrapper functions that deeply simplify the workflow for the user.

However, these functions come with a tree of parameters that have been predefined by the developers.
Variability during data acquisition (subject, acquisition system) can affect the recorded signals.
Such variability may make some predefined parameters of :py:mod:`physio` inappropriate. 
 
In this situation, we encourage the user to fine tune certain parameters by re-assigning values to the keys of the `parameters` dictionary 
got from the :py:func:`physio.get_ecg_parameters` or :py:func:`physio.get_respiration_parameters`. 
**To fine-tune parameters properly, a good understanding of each parameter's role is required...** and this why we wrote this section. 

Organization
------------

This tutorial is organized into two major sections, each one divided in multiple ones:
  * Respiration parameters
     - sentor_type = airflow
        - preprocess
        - smooth
        - cycle_detection
        - baseline
        - cycle_clean
     - sentor_type = belt
        - preprocess
        - smooth
        - cycle_detection
        - baseline
        - cycle_clean
     - sentor_type = co2
        - preprocess
        - smooth
        - cycle_detection
        - baseline
        - cycle_clean
  * ECG parameters
     - overview of the controllable parameters
     - preprocess
     - peak_detection
     - peak_clean



1) Respiration Parameters
-------------------------

For now, we have developed capabilities in :py:mod:`physio` to process respiration recorded with three types of sensors: `airflow`, `belt`, and `co2`.  
The sensor type drives many of the subsequent parameters, from preprocessing to metric computation, and therefore affects which metrics can be computed.  

This section will detail the parameter settings for each of these sensor types.




a. `airflow`

Default parameters dictionary for `airflow` sensor:
::

    {
        'baseline': {
            'baseline_mode': 'median'
        },
        'cycle_clean': {
            'low_limit_log_ratio': 4.5,
            'variable_names': ['inspi_volume', 'expi_volume']
        },
        'cycle_detection': {
            'epsilon_factor1': 10.0,
            'epsilon_factor2': 5.0,
            'inspiration_adjust_on_derivative': False,
            'method': 'crossing_baseline'
        },
        'preprocess': {
            'band': 7.0,
            'btype': 'lowpass',
            'ftype': 'bessel',
            'normalize': False,
            'order': 5
        },
        'sensor_type': 'airflow',
        'smooth': {
            'sigma_ms': 60.0,
            'win_shape': 'gaussian'
        }
    }

- `preprocess`:

The `preprocess` key controls how the raw respiratory signal is filtered. This is done using `scipy.signal.iirfilter` (see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirfilter.html).  
Several subkeys relate to controlling this filtering:

  - `btype`: The type of filter. In this context, we set a `lowpass`, but it could be `bandpass` to remove slow drifts by setting low and high cutoffs.  
  - `band`: The cutoff frequency. For a `lowpass`, this is the high cutoff, set to 7 Hz by default. **This parameter strongly affects the precise timing of respiratory cycle timepoint detection.** Decreasing this value increases signal smoothness but may artificially shift the inspiration-expiration transition.  
  - `ftype`: The type of filter. For example, `bessel` (default) or `butter`. We recommend `bessel` because it preserves time-domain fidelity, although it is slightly less steep in frequency cutoff.  
  - `order`: The filter order. Default = 5. Higher order → steeper frequency cutoff but increases the risk of phase distortion.  
  - `normalize`: True or False. If True, the signal is normalized by subtracting its mean and dividing by its MAD (Median Absolute Deviation). Default = False. Useful to scale the respiratory signal into a "normal" range, for example, to compare it to another normalized signal.  

- `smooth`:

The `smooth` key controls how the filtered respiratory signal is smoothed again using convolution with a kernel.  
Subkeys relate to the size and shape of this kernel:

  - `win_shape`: Default = `gaussian`. The kernel shape is Gaussian. It can be set to `rect` for a rectangular kernel, but we recommend `gaussian` because it reduces discontinuities in time, even if its frequency response is less steep than `rect`.  
  - `sigma_ms`: Kernel size in milliseconds. Higher → smoother; lower → less smooth. Default = 60 ms.  

- `cycle_detection`:

This key controls how the main timepoints (inspiration and expiration) are detected:

  - `method`: `crossing_baseline` in this case, because the signal returns to baseline when there is no flow (unlike belt and CO2 signals, where the method is set to `min_max`).  
  - `epsilon_factor1`: Defines a horizontal confidence zone just below the true baseline, where the low part = baseline - `epsilon` * `epsilon_factor1`, with `epsilon` = (baseline - np.quantile(resp, 0.1)) / 100. See Fig X.  
  - `epsilon_factor2`: Defines the higher part of the confidence zone: baseline - `epsilon` * `epsilon_factor2`. `epsilon_factor1` is higher than `epsilon_factor2` to search the low part of the confidence zone. See Fig X.  
  - `inspiration_adjust_on_derivative`: Sometimes the end-of-expiration plateau drifts downward, causing premature detection of inspiration. Activating this parameter adjusts detection using the slope's minimum (second derivative). Default = False.  

- `cycle_clean`:

This key controls how already detected timepoints are cleaned.  
Small oscillations in a noisy signal can cause very small cycles to be detected when the signal crosses the baseline. Subkeys specify criteria for identifying these outliers:

  - `variable_names`: Names of respiratory features used to detect outliers. Default = ['inspi_volume', 'expi_volume']. Volumes are chosen because they capture cycles that are too small both in time and amplitude.  
  - `low_limit_log_ratio`: Features are often non-normally distributed and are log-transformed before threshold estimation. The threshold for outliers is computed as median - MAD * `low_limit_log_ratio`. Higher `low_limit_log_ratio` → smaller detected cycles → fewer outliers detected. See Fig X.  

- `baseline`:

Controls how the baseline of the signal is computed:

  - `baseline_mode`: Default = `median`, meaning the baseline is the median level of the signal (robust and efficient). Alternatives: `zero` (baseline = 0) or `mode` (mode of the signal distribution).  

b. `belt`

Default parameters dictionary for `belt` sensor:

::

    {
        'baseline': None,
        'cycle_clean': {
            'low_limit_log_ratio': 8.0,
            'variable_names': ['inspi_amplitude', 'expi_amplitude']
        },
        'cycle_detection': {
            'exclude_sweep_ms': 200.0,
            'method': 'min_max'
        },
        'preprocess': {
            'band': 5.0,
            'btype': 'lowpass',
            'ftype': 'bessel',
            'normalize': False,
            'order': 5
        },
        'sensor_type': 'belt',
        'smooth': {
            'sigma_ms': 40.0,
            'win_shape': 'gaussian'
        }
    }


c. co2


1) ECG parameters
-------------------------

blablabla