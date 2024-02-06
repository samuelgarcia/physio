import numpy as np
import scipy.interpolate

from tqdm.auto import tqdm


def deform_traces_to_cycle_template(data, times, cycle_times, points_per_cycle=40,
                                    segment_ratios=None, output_mode='stacked', progress_bar=False):
    """

    Deform a signal a 1D signal (or also ND) to a 'cycle template'.
    Every cycle chunk in the signal will be interpolated to have the same length.
    In short it is a times to cycles transformation.

    Parameters
    ----------

    data:
        ND array time axis must always be 0
    times:
        real timestamps associated to data
    cycle_times: np.array
        Array with shape(num_cycle, num_segment + 1).
        Typically for respiration n cycles array with 3 columns (inspi time + expi time + next inspi time) will
        make deformation with 2 segments.
        If the cycle_times is 1D then it is converted to shape (size-1, 2).
        The end of every cycles must match the start of the next cycle.
    points_per_cycle: int (default 40)
        number of respi phase per cycle
    segment_ratios: None or float or list of float
        If multi segment deformation then a list of segmetn ratio must provived.
    output_mode: 'stacked' / 'unstacked' / 'unstacked_full'

    Returns
    -------
    If mode == 'stacked'
    
    deformed_data_stacked: 
        A 2d array of stacked cycles. Shape = (num_cycles, points_per_cycle)

    If mode == 'unstacked'

    deformed_data: 
        A 1d array of deformed cycles. Shape = (num_cycles * points_per_cycle)
    cycle_points: 
        The cycle vector

    If mode == 'unstacked':
    clipped_times: 
        The clipped time vector
    times_to_cycles:
        The vector times to cycle
    """

    if cycle_times.ndim == 1:
        cycle_times_1d = cycle_times
        cycle_times = np.zeros((cycle_times_1d.size - 1, 2), dtype=cycle_times_1d.dtype)
        cycle_times[:, 0] = cycle_times_1d[:-1]
        cycle_times[:, 1] = cycle_times_1d[1:]

    # check that the end of a cycle is the same the start of the following cycle
    assert (cycle_times[1:, 0] == cycle_times[:-1, -1]).all(), 'Start and end cycle times do not match'
    
    num_seg_phase = cycle_times.shape[1] - 1

    if num_seg_phase == 1:
        assert segment_ratios is None
        ratios = [0., 1.]
    else:
        assert segment_ratios is not None
        if num_seg_phase == 2 and np.isscalar(segment_ratios):
            segment_ratios = [segment_ratios]
        assert len(segment_ratios) == num_seg_phase - 1
        ratios = [0.] + list(segment_ratios) + [1.]

    # TODO check if this work with more segment
    assert num_seg_phase in (1, 2)

    # check that cycle_times are inside time range
    t_start, t_stop = times[0], times[-1]
    assert cycle_times[0, 0] >= t_start
    assert cycle_times[-1, -1] <= t_stop
    # keep_cycles = (cycle_times[:, 0] >= t_start) & (cycle_times[:, -1] < t_stop)
    # cycles_inds, = np.nonzero(keep_cycles)

    # clip times/data outside cycle_times range
    keep_times = (times >= cycle_times[0, 0]) & (times < cycle_times[-1, -1])
    clipped_times = times[keep_times]
    clipped_data = data[keep_times]


    # construct cycle_step
    times_to_cycles = np.full(clipped_times.shape, np.nan)
    times_to_cycles = np.full(clipped_times.shape, np.nan)
    loop = range(cycle_times.shape[0])
    if progress_bar:
        loop = tqdm(loop)

    for c in loop:
        for s in range(num_seg_phase):
            # mask_times = (clipped_times >= cycle_times[c, s]) & (clipped_times < cycle_times[c, s+1])
            # inds, = np.nonzero(mask_times)
            
            i0 = np.searchsorted(clipped_times, cycle_times[c, s])
            i1 = np.searchsorted(clipped_times, cycle_times[c, s+1])
            # print(inds[0], inds[-1], i0, i1)

            times_to_cycles[i0:i1] = (clipped_times[i0:i1] - cycle_times[c, s]) / \
                                          (cycle_times[c, s + 1] - cycle_times[c, s]) * (ratios[s + 1] - ratios[s]) + \
                                          c + ratios[s]

    interp = scipy.interpolate.interp1d(times_to_cycles, clipped_data, kind='linear', axis=0, bounds_error=False, fill_value='extrapolate')
    # cycle_points = np.arange(cycles_inds[0], cycles_inds[-1] + 1, 1. / points_per_cycle)
    cycle_points = np.arange(0, cycle_times.shape[0], 1. / points_per_cycle)
    deformed_data = interp(cycle_points)

    if output_mode == 'stacked':
        shape = (cycle_times.shape[0], points_per_cycle)
        if data.ndim > 1:
            shape =shape  + data.shape[1:]
        deformed_data_stacked = deformed_data.reshape(*shape)
        return deformed_data_stacked
    elif output_mode == 'unstacked':
        return deformed_data, cycle_points
    elif output_mode == 'unstacked_full':
        return deformed_data, cycle_points, clipped_times, times_to_cycles
    else:
        raise ValueError(f'Wrong output_mode {output_mode}')



def time_to_cycle(times, cycle_times,  segment_ratios = 0.4):
    """
    Map absolut event time to cycle position.
    Useful for event to respiration cycle histogram
    
    Parameters
    ----------

    segment_ratios: None or float or list of float
        If multi segment deformation then a list of segmetn ratio must provived.

    Returns
    -------

    """
    if cycle_times.ndim == 1:
        cycle_times_1d = cycle_times
        cycle_times = np.zeros((cycle_times_1d.size - 1, 2), dtype=cycle_times_1d.dtype)
        cycle_times[:, 0] = cycle_times_1d[:-1]
        cycle_times[:, 1] = cycle_times_1d[1:]

    # check that the end of a cycle is the same the start of the following cycle
    assert (cycle_times[1:, 0] == cycle_times[:-1, -1]).all(), 'Start and end cycle times do not match'
    
    num_seg_phase = cycle_times.shape[1] - 1

    if num_seg_phase == 1:
        assert segment_ratios is None
        ratios = [0., 1.]
    else:
        assert segment_ratios is not None
        if num_seg_phase == 2 and np.isscalar(segment_ratios):
            segment_ratios = [segment_ratios]
        assert len(segment_ratios) == num_seg_phase - 1
        ratios = [0.] + list(segment_ratios) + [1.]





    n = cycle_times.shape[0]

    # num_seg_phase = cycle_times.shape[1]
    # assert num_seg_phase in (1, 2)
    
    
    cycle_point = np.zeros((cycle_times.shape[0], len(ratios) - 1))
    for i in range(len(ratios) - 1):
        cycle_point[:, i] = np .arange(n) + ratios[i]

    
    flat_cycle_times = cycle_times[:, :-1].flatten()
    flat_cycle_point = cycle_point.flatten()
    keep = ~np.isnan(flat_cycle_times)
    flat_cycle_times = flat_cycle_times[keep]
    flat_cycle_point = flat_cycle_point[keep]
    interp = scipy.interpolate.interp1d(flat_cycle_times, flat_cycle_point, kind='linear', axis=0, bounds_error=False, fill_value='extrapolate')
    
    
    inside = (times>=cycle_times[0,0]) & (times<cycle_times[-1,0])
    cycles = np.zeros_like(times) * np.nan
    cycles[inside] = interp(times[inside])
    
    # put nan when some times are in missing cycles
    if num_seg_phase == 2:
        ind_missing, = np.nonzero(np.isnan(cycle_times[:, 1]))
        in_missing = np.in1d(np.floor(cycles), ind_missing.astype(cycles.dtype))
        cycles[in_missing] = np.nan
    
    
    return cycles
    
