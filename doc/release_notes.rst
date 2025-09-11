.. _releasenotes:

=============
Release notes
=============


Version 0.3.0
=============

2025-09-11

  * Doc massive improvement
  * Add sensor_type concept for respiration 
  * Add sensor_type=belt with min_max method for cycle detection
  * Refactor the clean_respiration_cycles() for cleaning small cycles
  * Rename compute_rsa() to compute_resphrv()
  * Refine parameters and add new presets


Version 0.2.0
=============

2024-01-06

  * improve respiration cycle detection (2 methods : crossing baseline and co2)
  * more doc
  * proof of concept of cardio-respiration synchronisation


Version 0.1.0
=============

2023-06-09

**Initial Version with**:

  * respiration cycle detection
  * respiration cycle features (amplitude, duration, volumes ...)
  * simple preprocess on signal : filter using scipy and smoothing
  * ecg peak detection
  * ecg/hrv metrics (time domain and frequency domain)
  * rsa : new approach to get cycle-by-cycle metrics
  * cyclic deformation machinery : a simple stretcher of any signal to cycle template
  * simple reader of micromed and brainvision using neo
  * "auto-magic" parameters for different species
