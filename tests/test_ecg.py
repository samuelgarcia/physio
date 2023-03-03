import numpy as np
from pathlib import Path

from physio import compute_ecg, compute_ecg_metrics, compute_instantaneous_rate

# read signals
example_folder = Path(__file__).parents[1] / 'examples'
raw_ecg = np.load(example_folder / 'ecg1.npy')
srate = 1000.



def test_ecg():
    ecg, ecg_peaks = compute_ecg(raw_ecg, srate)

    compute_ecg_metrics(ecg_peaks, srate, min_interval_ms=500.,
                        max_interval_ms=2000., verbose = False)


    # ecg rate sampled at 10Hz
    times = np.arange(0, raw_ecg.size / srate, 0.01)
    peak_times = ecg_peaks / srate
    rate = compute_instantaneous_rate(peak_times, times, limits=None, units='bpm', interpolation_kind='linear')
    # print(rate)


if __name__ == '__main__':
    test_ecg()

