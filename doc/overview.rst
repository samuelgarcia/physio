Physio package overview
=======================

What can be done with the Physio toolbox?
-----------------------------------------

**Features**:

  * respiration cycle detection
  * respiration cycle features (amplitude, duration, volumes, …)
  * simple preprocessing on signals: filtering with SciPy and smoothing
  * ECG peak detection
  * ECG/HRV metrics (time domain and frequency domain)
  * RespHRV: a new approach to compute cycle-by-cycle metrics
  * cyclic deformation machinery: a simple stretcher of any signal to a cycle template
  * simple readers for Micromed and BrainVision using Neo
  * “auto-magic” parameters for different species


Why another Python toolbox for ECG and respiration?
---------------------------------------------------

The :py:mod:`physio` Python toolbox was developed in the  
`CMO <https://www.crnl.fr/fr/equipe/cmo>`_ (Codage Olfaction Mémoire) team of the Centre de Recherche en Neuroscience de Lyon (CRNL).

The **CMO** team has long-term experience analyzing respiration and neural events. See, for example, the 2006 paper:  
`Respiratory cycle as time basis: An improved method for averaging olfactory neural events <https://pubmed.ncbi.nlm.nih.gov/16246424/>`_

Many parts of our code for analyzing respiration were never published or released as open source, despite our strong contributions to many open projects (Neo, SpikeInterface, EphyViewer).

Some methods (e.g., cyclic deformation on respiratory signals) are home-made and central to many of our analyses. Making them public could be useful to others.

Finally, regarding RespHRV (Respiratory Heart Rate Variability, formerly RSA), we truly believe that respiration and ECG must be analyzed jointly. The development of this toolbox has been mainly motivated by this simple idea.

Parameters handling
-------------------

Detecting respiratory and ECG cycles involves many nested parameters: filtering, smoothing, thresholds, …  
These parameters strongly impact the results and are entirely species/protocol/state dependent. They also depend on the sensor used (e.g., nasal airflow vs. inductive belt vs. plethysmograph).

:py:mod:`physio` comes with predefined parameter presets for a few species (human, rodent) and sensors.  
By default, these presets should work without too much effort. For better results, they can be fine-tuned to improve cycle detection.


Design choices
--------------

* simplicity: easy to read, easy to understand, easy to hack
* functions only (no custom classes or complicated structures); input/output are NumPy arrays or Pandas DataFrames
* frugality: only a few essential features
* few dependencies: NumPy, Pandas, SciPy, and Neo (for data reading)


Comparison to other toolboxes
-----------------------------

:py:mod:`physio` is not the first open-source toolbox to analyze ECG and respiratory signals!  
Here is a very biased and unfair comparison with some other well-known toolboxes used in many academic labs.  
*If you are the author of one of these toolboxes: first, many thanks for sharing your code, and apologies for this comment. Feel free to give us feedback on ours!*

* **neurokit2** <https://neuropsychology.github.io/NeuroKit>:  
  A very impressive, generalist, and widely used Python toolbox for analyzing neurophysiological signals (EEG, respiration, ECG, EMG, EDA).  
  However, the accuracy of respiratory cycle detection and ECG R-peak detection was not sufficient for our needs without extensive modification.  
  We also find the code difficult to follow and hack due to deeply nested functions and complex data structures.

* **py-ecg-detectors** <https://github.com/berndporr/py-ecg-detectors>:  
  One of the best implementations (to our knowledge) of many R-peak detectors.  
  However, this package does not handle respiration. Many of the algorithms also do not detect the exact position of R peaks (a limitation of the methods themselves, not the implementation).

* **biosppy** <https://biosppy.readthedocs.io/>:  
  Has an inconvenient dependency on Tinker and OpenCV.  
  Implements a good R-peak detector with the Hamilton method, but documentation is limited. Does not include RespHRV analysis.

* **pyhrv** <https://pyhrv.readthedocs.io/en/latest/>:  
  Depends on biosppy and inherits the same heavy dependency list.  
  ECG peak detection is handled by biosppy. ECG metrics (time and frequency domain) are similar to :py:mod:`physio`, but we prefer the Pandas DataFrame approach, which leads to more compact code.

* **breathmetrics** <https://github.com/zelanolab/breathmetrics>:  
  A toolbox mainly focused on respiration. Unfortunately, it still targets MATLAB users.  
  It makes strong assumptions about “pauses” in the respiratory signal, which is true in general, but the pause between inhalation and exhalation can be an inconvenient assumption.  
  Most respiratory cycle metrics are also covered by :py:mod:`physio`.  
  breathmetrics includes a MATLAB GUI, which does not align with the frugality approach of :py:mod:`physio`.


Cite
----

We have written a `manuscript <https://www.eneuro.org/content/10/10/ENEURO.0197-23.2023>`_ describing this toolbox and the cycle-by-cycle RespHRV method.

If you use this toolbox, a citation would be greatly appreciated.

You can also check some notebooks used to benchmark and test this toolbox  
`here <https://github.com/samuelgarcia/physio_benchmark>`_



Authors
-------

Samuel Garcia, CNRS, lab engineer

Valentin Ghibaudo, PostDoc in neurosciences

Jules Granget, PostDoc in neurosciences
