import numpy as np
from pathlib import Path

from physio import compute_respiration, compute_ecg, compute_resphrv, compute_resphrv_rate_period

# read signals
example_folder = Path(__file__).parents[1] / 'examples'
raw_resp = np.load(example_folder / 'resp1_airflow.npy')
raw_ecg = np.load(example_folder / 'ecg1.npy')
srate = 1000.





def test_compute_resphrv():
    
    resp, resp_cycles = compute_respiration(raw_resp, srate, parameter_preset='human_airflow')
    ecg, ecg_peaks = compute_ecg(raw_ecg, srate, parameter_preset='human_ecg')



    # resphrv_cycles, cyclic_cardiac_rate = compute_resphrv(resp_cycles, ecg_peaks, srate=10., two_segment=True)
    # print(resphrv_cycles)
    # print(resphrv_cycles['decay_amplitude'].describe())

    # resphrv_cycles, cyclic_cardiac_rate_1seg = compute_resphrv(resp_cycles, ecg_peaks, srate=5., two_segment=False)

    # resphrv_cycles = compute_resphrv_rate_period(resp_cycles, ecg_peaks, return_cyclic_cardiac_rate = False, return_cyclic_cardiac_period = False)
    # print(resphrv_cycles.describe())
    # print(resphrv_cycles['decay_amplitude_bpm'].describe())
    # print(resphrv_cycles[['decay_amplitude_bpm','rising_amplitude_ms']])
    # print(resphrv_cycles.isna().sum())

    # resphrv_cycles, cyclic_cardiac_rate = compute_resphrv_rate_period(resp_cycles, ecg_peaks, return_cyclic_cardiac_rate = True, return_cyclic_cardiac_period = False)
    # print(cyclic_cardiac_rate)
    # print(resphrv_cycles['decay_amplitude_bpm'].values)
    # print(np.ptp(cyclic_cardiac_rate, axis = 1))

    resphrv_cycles, cyclic_cardiac_rate, cyclic_cardiac_period = compute_resphrv_rate_period(resp_cycles, ecg_peaks, return_cyclic_cardiac_rate = True, return_cyclic_cardiac_period = True)
    print(resphrv_cycles)
    print(cyclic_cardiac_rate)
    # print(resphrv_cycles['decay_amplitude_bpm'].values)
    # print(np.ptp(cyclic_cardiac_rate, axis = 1))

    print(cyclic_cardiac_period)
    # print(resphrv_cycles['rising_amplitude_ms'].values)
    # print(np.ptp(cyclic_cardiac_period, axis = 1))
    # print(resphrv_cycles['average_period_ms'].values.round(0))
    # print(np.median(cyclic_cardiac_period, axis = 1).round(0))


    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(nrows=2)
    # axs[0].plot(cyclic_cardiac_rate.T, color='k')
    # axs[1].plot(cyclic_cardiac_rate_1seg.T, color='k')
    # plt.show()





if __name__ == '__main__':
    test_compute_resphrv()
