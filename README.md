# physio

Simple python toolbox to analyse physio signals (Respiration, ECG, and RSA)

**Please jump to documentation now https://physio.readthedocs.io**

## Manuscript

This work has been published at eNeuro : https://www.eneuro.org/content/10/10/ENEURO.0197-23.2023


## Features

  * respiration cycle detection
  * respiration cycle features (amplitude, duration, volumes ...)
  * simple preprocess on signal : filter using scipy and smoothing
  * ecg peak detection
  * ecg/hrv metrics (time domain and freq domain)
  * rsa : new approach to get cycle-per-cycle metrics
  * cyclic deformation machinery : a simple strecher of any signal to cycle template
  * simple reader of micromed and brainvision using neo
  * "automagic" parameters for differents species


## Installation from PyPi

```bash
pip install physio
```


## Installation from source

```bash
git clone https://github.com/samuelgarcia/physio.git
cd physio
pip install -e .
```

Update:
```bash
cd physio
git pull origin main
```

## Authors 

Samuel Garcia with the help of Valentin Ghibaudo and Jules Granget

This toolbox is used in the CMO team from the CRNL.
