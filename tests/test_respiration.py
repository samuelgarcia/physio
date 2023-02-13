import numpy as np
from pathlib import Path

from physio import compute_respiration, detect_respiration_cycles, preprocess

# read signals
test_folder = Path(__file__).parent
raw_resp = np.load(test_folder / 'resp1.npy')
srate = 1000.



def test_compute_respiration():
    
    resp, cycle_features = compute_respiration(raw_resp, srate)
    print(resp.shape)
    print(cycle_features.shape)
    print(cycle_features)


def test_detect_respiration_cycles():

    resp = preprocess(raw_resp, srate, band=25., btype='lowpass', ftype='bessel', order=5, normalize=False)


    cycles = detect_respiration_cycles(resp, srate, baseline_mode='median',
                              inspration_ajust_on_derivative=False)



    # clean_respiration_cycles


if __name__ == '__main__':
    test_compute_respiration()

    # test_detect_respiration_cycles()

