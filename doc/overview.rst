Overview
========



What can be done with physio toolbox ?
--------------------------------------

**Features**:

  * respiration cycle detection
  * respiration cycle features (amplitude, duration, volumns ...)
  * simple preprocess on signal : filter using scipy and smoothing
  * ecg peak detection
  * ecg metrics
  * rsa
  * cyclic deformation machinery : a simple strecher of any signal to cycle template
  * simple reader of micromed and brainvision using neo
  * "automagic" parameters for differents species



Why another python toolbox for ecg and respiration ?
----------------------------------------------------

physio python toolbox has been develop in the CMO team of Centre de Recherche en Neuroscience de Lyon (CRNL).

The [**CMO**](https://www.crnl.fr/fr/equipe/cmo) (Codage Olfaction Mémoire) team has a long term experience in analysing respiration and neural events.
See this 2006 paper
[Respiratory cycle as time basis: An improved method for averaging olfactory neural events](https://pubmed.ncbi.nlm.nih.gov/16246424/)

Many piece of our codes for analysing respiration were never published and released properly as open source despite
the fact we are higly contributing to many open source projects.

Also some methods (cyclic deformation on respiratory signal) is some kind of home made signature of many analyses from
the team, having a public tool could be usefull for others.

We truely believe that respiration and ecg must be analysed jointly



Parameters handling
-------------------

Detecting respiratory cycles and ECG cycles have many nested parameters  : filetering, smotthing, threshold, ...
This parameters have a high impact on the results and are totaly species/protocol/state dependent.
This parameters also depend on the sensor used (nasal airflow vs inductive belt for instance vs plethysmo)

physio come with some predefined parameters set for a few species (human, rodent) and sensor
By default this parameters set should work without too much pain. For better results they can be finely tuned to get
better results on cycle detection.

TODO some code examples.


Design choice
-------------

* simplicity: eays to read and easy to hack
* function only (no custum class or complicated structures). input/ouput are numpy arrays or pandas dataframe.
* frugality : few features
* few dependencies: numpy, pandas, scipy and neo (for data reading)





Comparison to other toolboxes
-----------------------------

* [**neurokit2**](https://neuropsychology.github.io/NeuroKit): A very impressive, generalist and widely used python
  toolbox to analyse neurophysiogical signals (EEG, resp, ECG, EMG, EDA).
  The quality of respiratory cycles detction and ECG rpeak detection was not accurate enough to be used without hacking.
  We get the feeling that code is really hard to follow and hack, (deep nested function and too 
  sophisticated data structure).

* [**py-ecg-detectors**](https://github.com/berndporr/py-ecg-detectors): One of the best implemention (to our knowledge)
  of many R peak detector. But this package do not handle respiration. And many the proposed algorithms do not detect_peak
  the extact position of the R peaks, this inherents to methods.


* [**biospy**](https://biosppy.readthedocs.io/)

* [**pyhrv**](https://pyhrv.readthedocs.io/en/latest/)


Cite
----

We are wrtting a short paper to describe this toolbox here : TODO put URL

If you use this toolbox a citation would be appreciated for sure.

You can also check some notebook used to benchmark and test this toolbox
[here](https://github.com/samuelgarcia/physio_benchmark)


Authors
-------

Samuel Garcia, CNRS, lab ingineer

Valentin Ghibaudo, neuroscience PhD student

