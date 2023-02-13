import numpy as np
from pathlib import Path

from physio import compute_respiration, deform_traces_to_cycle_template


# read signals
test_folder = Path(__file__).parent
raw_resp = np.load(test_folder / 'resp1.npy')
srate = 1000.


def test_deform_traces_to_cycle_template():
    
    # test with resp deform iself
    times = np.arange(raw_resp.size) / srate

    resp, cycle_features = compute_respiration(raw_resp, srate)

    # one segment
    cycle_times = cycle_features[['start_time', 'stop_time']].values
    clipped_times, times_to_cycles, cycle_inds, cycle_points, deformed_resp = \
        deform_traces_to_cycle_template(raw_resp, times, cycle_times, points_per_cycle=40, segment_ratios=None)
    print(cycle_times.shape, cycle_inds.shape)

    # two segments
    cycle_times = cycle_features[['start_time', 'expi_time', 'stop_time']].values
    clipped_times, times_to_cycles, cycle_inds, cycle_points, deformed_resp = \
        deform_traces_to_cycle_template(raw_resp, times, cycle_times, points_per_cycle=40, segment_ratios=0.4)
    print(cycle_times.shape, cycle_inds.shape)


if __name__ == '__main__':
    test_deform_traces_to_cycle_template()
