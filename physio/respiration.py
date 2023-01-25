import numpy as np
from .tools import get_empirical_mode

def detect_respiration_cycles(resp, srate, baseline_mode='manual', baseline=None, ):
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

    # assert 

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