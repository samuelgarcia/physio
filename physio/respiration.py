import numpy as np
from .tools import get_empirical_mode

def detect_respiration_cycles(resp, srate, baseline_mode='manual', baseline=None,  inspration_ajust_on_derivative=False):
    """
    Detect respiration cycles based on:
      * crossing zeros (or crossing baseline)
      * some cleanning with euristicts

    
    """

    if baseline_mode == 'manual':
        assert baseline is not None
    elif baseline_mode == 'zero':
        baseline = 0.
    elif baseline_mode == 'median':
        baseline = np.median(resp)
    elif baseline_mode == 'mode':
        baseline = get_empirical_mode(resp)

    resp0 = resp[:-1]
    resp1 = resp[1:]

    ind_insp, = np.nonzero((resp0 >= baseline) & (resp1 < baseline))
    ind_exp, = np.nonzero((resp0 < baseline) & (resp1 >= baseline))

    if ind_insp.size == 0:
        print('no cycle dettected')
        return
    
    mask = (ind_exp > ind_insp[0]) & (ind_exp < ind_insp[-1])
    ind_exp = ind_exp[mask]

    print(ind_insp.size, ind_exp.size)
    
    if inspration_ajust_on_derivative:
        # lets find local minima on second derivative
        # this can be slow
        delta_ms = 10.
        delta = int(delta_ms * srate / 1000.)
        derivate1 = np.gradient(resp)
        derivate2 = np.gradient(derivate1)
        for i in range(ind_exp.size):
            i0, i1 = ind_insp[i], ind_exp[i]
            i0 = max(0, i0 - delta)
            i1 = i0 + np.argmin(resp[i0:i1])
            d1 = derivate1[i0:i1]
            i1 = i0 + np.argmin(d1)
            if (i1 - i0) >2:
                # find the last crossing zeros in this this short segment
                d2 = derivate2[i0:i1]
                i1 = i0 + np.argmin(d2)
                if (i1 - i0) >2:
                    d2 = derivate2[i0:i1]
                    mask = (d2[:-1] >=0) & (d2[1:] < 0)
                    if np.any(mask):
                        ind_insp[i] = i0 + np.nonzero(mask)[0][-1]
                
            
            #~ import matplotlib.pyplot as plt
            #~ fig, axs = plt.subplots(nrows=3, sharex=True)
            #~ i1 = ind_exp[i]
            #~ axs[0].plot(resp[i0: i1])
            #~ l = ind_insp[i] - i0
            #~ axs[0].axvline(l)
            #~ axs[1].plot(derivate1[i0:i1])
            #~ axs[1].plot(d1)
            #~ axs[1].axvline(l)
            #~ axs[2].plot(derivate2[i0:i1])
            #~ axs[2].plot(d2)
            #~ axs[2].axvline(l)
            #~ axs[2].axhline(0)
            #~ plt.show()

    cycles = np.zeros((ind_insp.size, 2), dtype='int64')
    cycles[:, 0] = ind_insp
    cycles[:-1, 1] = ind_exp
    cycles[-1, 1] = -1


    return cycles


    # return cycles



def compute_respiration(raw_resp, srate):
    """
    
    """
    pass