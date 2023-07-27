.. _releasenotes:

=============
Release notes
=============




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
