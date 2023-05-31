import numpy as np
from pathlib import Path

from physio import compute_respiration, deform_traces_to_cycle_template


# read signals
example_folder = Path(__file__).parents[1] / 'examples'
raw_resp = np.load(example_folder / 'resp1.npy')
srate = 1000.


def test_deform_traces_to_cycle_template():
    
    # test with resp deform iself
    times = np.arange(raw_resp.size) / srate

    resp, resp_cycles = compute_respiration(raw_resp, srate)

    # one segment
    cycle_times = resp_cycles[['inspi_time', 'next_inspi_time']].values
    deformed_resp = deform_traces_to_cycle_template(raw_resp, times, cycle_times,
                                                    points_per_cycle=40, segment_ratios=None,
                                                    output_mode='stacked')
    assert deformed_resp.ndim == 2
    assert deformed_resp.shape[0] == cycle_times.shape[0]

    # two segments
    cycle_times = resp_cycles[['inspi_time', 'expi_time', 'next_inspi_time']].values
    deformed_resp = deform_traces_to_cycle_template(raw_resp, times, cycle_times,
                                                    points_per_cycle=40, segment_ratios=0.4,
                                                    output_mode='stacked')
    assert deformed_resp.ndim == 2
    assert deformed_resp.shape[0] == cycle_times.shape[0]

    # one segment 
    cycle_times = resp_cycles[['inspi_time', 'next_inspi_time']].values
    deformed_resp, cycle_points = deform_traces_to_cycle_template(raw_resp, times, cycle_times,
                                                    points_per_cycle=40, segment_ratios=None,
                                                    output_mode='unstacked')
    assert deformed_resp.ndim == 1
    assert deformed_resp.shape == cycle_points.shape


if __name__ == '__main__':
    test_deform_traces_to_cycle_template()
