API
===


respiration
-----------

.. automodule:: physio.respiration

    .. autofunction:: compute_respiration
    .. autofunction:: detect_respiration_cycles
    .. autofunction:: clean_respiration_cycles
    .. autofunction:: compute_respiration_cycle_features

ECG
---
    
.. automodule:: physio.ecg

    .. autofunction:: compute_ecg
    .. autofunction:: clean_ecg_peak
    .. autofunction:: compute_ecg_metrics
    .. autofunction:: compute_instantaneous_rate


reader
------
    
.. automodule:: physio.reader

    .. autofunction:: read_one_channel

preproces
---------
    
.. automodule:: physio.preprocess

    .. autofunction:: preprocess
    .. autofunction:: smooth_signal


plotting
--------
    
.. automodule:: physio.plotting

    .. autofunction:: plot_cyclic_deformation

tools
-----
    
.. automodule:: physio.tools

    .. autofunction:: compute_median_mad
    .. autofunction:: detect_peak
    .. autofunction:: get_empirical_mode
