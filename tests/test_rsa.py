import numpy as np
from pathlib import Path

from physio import compute_respiration, compute_ecg, compute_rsa

# read signals
example_folder = Path(__file__).parents[1] / 'examples'
raw_resp = np.load(example_folder / 'resp1.npy')
raw_ecg = np.load(example_folder / 'ecg1.npy')
srate = 1000.





def test_compute_rsa():
    
    resp, resp_cycles = compute_respiration(raw_resp, srate, parameter_preset='human_airflow')
    ecg, ecg_peaks = compute_ecg(raw_ecg, srate, parameter_preset='human_ecg')
    
    

    rsa_cycles, cyclic_cardiac_rate = compute_rsa(resp_cycles, ecg_peaks, srate=10., two_segment=True)
    print(rsa_cycles)

    rsa_cycles, cyclic_cardiac_rate_1seg = compute_rsa(resp_cycles, ecg_peaks, srate=5., two_segment=False)


    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(nrows=2)
    # axs[0].plot(cyclic_cardiac_rate.T, color='k')
    # axs[1].plot(cyclic_cardiac_rate_1seg.T, color='k')
    # plt.show()





if __name__ == '__main__':
    test_compute_rsa()
