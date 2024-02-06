import numpy as np
from pathlib import Path

from physio import compute_respiration, detect_respiration_cycles, preprocess

# read signals
example_folder = Path(__file__).parents[1] / 'examples'
raw_resp = np.load(example_folder / 'resp1.npy')
srate = 1000.



def test_compute_respiration():
    
    resp, resp_cycles = compute_respiration(raw_resp, srate, parameter_preset='human_airflow')
    # print(resp.shape)
    # print(resp_cycles.shape)
    # print(resp_cycles)
    # print(resp_cycles.dtypes)



def test_detect_respiration_cycles():

    resp = preprocess(raw_resp, srate, band=25., btype='lowpass', ftype='bessel', order=5, normalize=False)


    cycles = detect_respiration_cycles(resp, srate, method="crossing_baseline", baseline_mode='median',
                              inspiration_adjust_on_derivative=False)
    print(cycles.shape)


    # clean_respiration_cycles


if __name__ == '__main__':
    test_compute_respiration()

    test_detect_respiration_cycles()

