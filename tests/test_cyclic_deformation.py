import numpy as np
from pathlib import Path

from physio import compute_respiration, deform_traces_to_cycle_template, compute_ecg, time_to_cycle


# read signals
example_folder = Path(__file__).parents[1] / 'examples'
raw_resp = np.load(example_folder / 'resp1_airflow.npy')
raw_ecg = np.load(example_folder / 'ecg1.npy')
srate = 1000.


def test_deform_traces_to_cycle_template():
    
    # test with resp deform iself
    times = np.arange(raw_resp.size) / srate

    resp, resp_cycles = compute_respiration(raw_resp, srate, parameter_preset='human_airflow')

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



def test_time_to_cycle():
    _, resp_cycles = compute_respiration(raw_resp, srate, parameter_preset='human_airflow')
    _ , ecg_peaks = compute_ecg(raw_ecg, srate, parameter_preset = 'human_ecg')
    inspi_ratio = resp_cycles['cycle_ratio'].median()
    cycle_times = resp_cycles[['inspi_time', 'expi_time', 'next_inspi_time']].values
    rpeak_phase = time_to_cycle(ecg_peaks['peak_time'].values, cycle_times, segment_ratios=[inspi_ratio])


if __name__ == '__main__':
    # test_deform_traces_to_cycle_template()
    test_time_to_cycle()
