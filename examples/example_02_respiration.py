'''
Respiration example
===================


'''

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import physio

##############################################################################
# 
# physio.respiration
# ------------------
#
# 
#  

##############################################################################
# 
# Respiration : quick way
# -----------------------
#


raw_resp = np.load('resp1.npy')
srate = 1000.
times = np.arange(raw_resp.size) / srate

fig, ax = plt.subplots()
ax.plot(times, raw_resp)
ax.set_ylabel('raw resp')

ax.set_xlim(350, 450)


##############################################################################
# 
# Respiration : step by step
# --------------------------
#



plt.show()