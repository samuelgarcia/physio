documentation of physio 
=======================


:py:mod:`physio` is a python toolbox to analyse physiological signals : respiration and ECG

Author : Samuel Garcia with the help of Valentin Ghibaudo and Jules Granget.

This toolbox is used by the CMO team in the **Centre de recherche en Neurosciences de Lyon (CRNL)**


**Features** :

  * respiration cycle detection
  * respiration cycle features (amplitude, duration, volumns ...)
  * simple preprocess on signal : filter using scipy and smoothing
  * ecg peak detection
  * ecg metrics
  * cyclic deformation machinery : a simple strecher of any signal to cycle template
  * simple reader of micromed and brainvision using neo



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



.. toctree::
   :maxdepth: 1
   :caption: Contents:

   api
   release_notes



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
