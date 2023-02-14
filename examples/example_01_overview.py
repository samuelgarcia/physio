'''
physio overview
===============

Here a quick overview of the :py:mod:`physio`
'''

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import physio

##############################################################################
# 
# :py:mod:`physio` have a util function for reading some format like:
#  
#    * micromed
#    * brainvision
#
# Here we use internal file in numpy format for the demo

raw_resp = np.load('resp1.npy')
raw_ecg = np.load('ecg1.npy')
srate = 1000.
times = np.arange(raw_resp.size) / srate

fig, axs = plt.subplots(nrows=2, sharex=True)
ax = axs[0]
ax.plot(times, raw_resp)
ax.set_ylabel('raw resp')

ax = axs[1]
ax.plot(times, raw_ecg)
ax.set_ylabel('raw ECG')

ax.set_xlim(185, 225)


##############################################################################
# 
# Analyse resp
# ------------
#  
# :py:func:`~physio.compute_respiration` is an easy function to:
#
#    * preprocess the respiration signal
#    * compute cycle
#    * compute cycle features


resp, cycle_features = physio.compute_respiration(raw_resp, srate)

fig, ax = plt.subplots()
ax.plot(times, raw_resp)
ax.plot(times, resp)
ax.set_xlim(185, 225)

##############################################################################
# 
# repiration cycles and features
# ------------------------------
#  
# 
#
# 
# 
# 

print(cycle_features.shape)
print(cycle_features.columns)

cycle_features


##############################################################################
# 

inspi_ind = cycle_features['inspi_index'].values
expi_ind = cycle_features['expi_index'].values

fig, ax = plt.subplots()
ax.plot(times, resp)
ax.scatter(times[inspi_ind], resp[inspi_ind], color='green')
ax.scatter(times[expi_ind], resp[expi_ind], color='red')
ax.set_xlim(185, 225)



##############################################################################
# 


plt.show()


