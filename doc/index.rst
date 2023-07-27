Documentation of toolbox physio
==============================


:py:mod:`physio` is a python toolbox to analyse physiological signals : respiration and ECG


Authors
-------

:py:mod:`physio` is developped by Samuel Garcia, Valentin Ghibaudo and Jules Granget.

This toolbox is used by the CMO team in the **Centre de recherche en Neurosciences de Lyon (CRNL)**



Preprint manuscript
-------------------

Have a look to the `preprint manuscript <https://osf.io/qbuzy/>`_ for more informations.



Features
--------

  * respiration cycle detection
  * respiration cycle features (amplitude, duration, volumes ...)
  * simple preprocess on signal : filter using scipy and smoothing
  * ecg peak detection
  * ecg/hrv metrics (time domain and frequency domain)
  * rsa : new approach to get cycle-by-cycle metrics
  * cyclic deformation machinery : a simple stretcher of any signal to cycle template
  * simple reader of micromed and brainvision using neo
  * "auto-magic" parameters for different species


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   overview
   api
   release_notes

.. include:: examples/index.rst


Installation
------------

Installation from pypi:

.. code-block:: bash

   pip install physio


Installation from sources:

.. code-block:: bash

   git clone https://github.com/samuelgarcia/physio.git
   cd physio
   pip install -e .

Update from source:

.. code-block:: bash

   cd physio
   git pull origin main







Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
