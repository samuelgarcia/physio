Physio package overview
=======================


What can be done with physio toolbox ?
--------------------------------------

**Features**:

  * respiration cycle detection
  * respiration cycle features (amplitude, duration, volumes ...)
  * simple preprocess on signal : filter using scipy and smoothing
  * ecg peak detection
  * ecg/hrv metrics (time domain and frequency domain)
  * rsa : new approach to get cycle-by-cycle metrics
  * cyclic deformation machinery : a simple stretcher of any signal to cycle template
  * simple reader of micromed and brainvision using neo
  * "auto-magic" parameters for different species


Why another python toolbox for ECG and respiration ?
----------------------------------------------------

:py:mod:`physio` python toolbox has been develop in the `CMO <https://www.crnl.fr/fr/equipe/cmo>`_
(Codage Olfaction Mémoire) team of Centre de Recherche en Neuroscience de Lyon (CRNL).

The  **CMO** team has a long term experience in analysing respiration and neural events. See this 2006 paper
`Respiratory cycle as time basis: An improved method for averaging olfactory neural events <https://pubmed.ncbi.nlm.nih.gov/16246424/>`_

Many pieces of our codes for analysing respiration were never published and not released properly as open source despite
the fact that we are highly contributing to many open source projects (neo, spikeinterface, ephyviewer).

Some methods (cyclic deformation on respiratory signal) is home made and very central for many analyses in
the team, having a public tool could be useful for others.

And finally, about the RSA (Respiratory Sinus Arrhythmia), we truely believe that respiration and ecg must be analysed
jointly, the developemment of this toolbox have been mainly motivated by this simple idea.


Parameters handling
-------------------

Detecting respiratory cycles and ECG cycles have many nested parameters : filtering, smoothing, threshold, ...
Theses parameters have a high impact on the results and are totaly species/protocol/state dependent.
Theses parameters also depend on the sensor used. For instance  : nasal airflow vs inductive belt vs plethysmo.

:py:mod:`physio` come with some predefined parameters preset for a few species (human, rodent) and sensors.
By default theses parameters preset should work without too much pain. For better results they can be finely tuned to get
better results on cycle detection.


Design choice
-------------

* simplicity: easy to read, easy to understand, easy to hack
* function only (no custom class or complicated structures). input/ouput are numpy arrays or pandas dataframe.
* frugality : few features
* few dependencies: numpy, pandas, scipy and neo (for data reading)


Comparison to other toolboxes
-----------------------------

:py:mod:`physio` is not the first toolbox on the open-source market to analyse ECG and respiratory signals!
Here a very biased and unfair comparison to some other well-known toolboxes used by many accademic labs.
*If you are one author of theses toolboxes, first many thanks for sharing your code and apologize for this comment.*
*Feel free to comment on our toolbox*


* `**neurokit2** <https://neuropsychology.github.io/NeuroKit>`_: A very impressive, generalist and widely used python
  toolbox to analyse neurophysiogical signals (EEG, resp, ECG, EMG, EDA).
  The quality of respiratory cycles detction and ECG rpeak detection was not accurate enough to be used without hacking
  to analyse our data.
  Also, we get the feeling that code is really hard to follow and hack (deep nested function and too 
  sophisticated data structure).

* `**py-ecg-detectors** <https://github.com/berndporr/py-ecg-detectors>`_: One of the best implemention (to our knowledge)
  of many R peak detector. But this package do not handle respiration. And many the proposed algorithms do not detect 
  the extact position of the R peaks  (this inherents to methods and not the implemention!).

* `**biospy** <https://biosppy.readthedocs.io/>`_ : this has a very annoying dependency to tinker and opencv.
  Implement a godd R peak detector with Hamilton method. Have limited documentation. Do not have RSA analyse.

* `**pyhrv** <https://pyhrv.readthedocs.io/en/latest/>`_: depend on biospy and have the same annoying dependencies list.
  The ECG peak is done on biospy. The ECG metrics (time and freq domain) are similar :py:mod:`physio` but we prefer
  the pandas DataFrame approach, this lead to much comptact code.

* `breathmetrics <https://github.com/zelanolab/breathmetrics>`_ : a toolbox mainly around the respiration. Unfortunatly
  still target matlab users. Have strong assumtion about "pauses" in the respiratory signal (which is true) but the pause
  between inhalation and exhalation can be somehow an annoying assumption. Most of respiratory cycles metrics are also
  handle by :py:mod:`physio`. breathmetrics have GUI written in matlab which do not embrace the frugality
  approach of :py:mod:`physio`


Cite
----

We have writted a `manuscript <https://www.eneuro.org/content/10/10/ENEURO.0197-23.2023>`_ to describe this toolbox
and the cycle-by-cycle RSA method.

If you use this toolbox a citation would be appreciated for sure.

You can also check some notebook used to benchmark and test this toolbox
`here <https://github.com/samuelgarcia/physio_benchmark>`_



Authors
-------

Samuel Garcia, CNRS, lab ingineer

Valentin Ghibaudo, neuroscience PhD student

Jules Granget, neuroscience PhD student

