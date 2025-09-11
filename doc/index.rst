Documentation of toolbox physio
==============================


:py:mod:`physio` is a python toolbox to analyse physiological signals : respiration and ECG (electrocardiogram).


Authors
-------

:py:mod:`physio` is developped by Samuel Garcia, Valentin Ghibaudo and Jules Granget.

This toolbox is used by the CMO team in the **Centre de recherche en Neurosciences de Lyon (CRNL)**



Manuscript
----------

This work has been published at eNeuro : https://www.eneuro.org/content/10/10/ENEURO.0197-23.2023


Features
--------

  * respiration cycle detection
  * respiration cycle features (amplitude, duration, volumes, …)
  * simple preprocessing on signals: filtering with SciPy and smoothing
  * ECG peak detection
  * ECG/HRV metrics (time domain and frequency domain)
  * RespHRV: a new approach to compute cycle-by-cycle metrics
  * cyclic deformation machinery: a simple stretcher of any signal to a cycle template
  * simple readers for Micromed and BrainVision using Neo
  * “auto-magic” parameters for different species (human, rat)


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   overview
   api
   handling_parameters
   release_notes

.. include:: examples/index.rst


Installation
------------

Installation from pypi (releases only):

.. code-block:: bash

   pip install physio


Installation from (github) sources (more up-to-date):

.. code-block:: bash
   
   pip install https://github.com/samuelgarcia/physio/archive/main.zip


Installation with local clone of sources (for dev mode):

.. code-block:: bash

   git clone https://github.com/samuelgarcia/physio.git
   cd physio
   pip install -e .

Update local sources:

.. code-block:: bash

   cd physio
   git pull origin main



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
