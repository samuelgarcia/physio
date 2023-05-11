import numpy as np
from pathlib import Path

from physio import compute_ecg, compute_ecg_metrics, compute_instantaneous_rate, compute_hrv_psd

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

    # compute LF/HF
    ecg_duration_s = raw_ecg.shape[0] / srate
    freqency_bands = {'lf': (0.04, .15), 'hf' : (0.15, .4)}
    psd_freqs, psd, metrics = compute_hrv_psd(peak_times, ecg_duration_s,  sample_rate=100., limits=None, units='bpm',
                                    freqency_bands=freqency_bands, window_s=500.)
    print(metrics)
    #~ import matplotlib.pyplot as plt
    #~ fig, axs = plt.subplots(nrows=2, sharex=True)
    #~ ax = axs[0]
    #~ ax.plot(psd_freqs, psd)
    #~ ax.set_title(str(metrics))
    #~ ax = axs[1]
    #~ ax.semilogy(psd_freqs, psd)
    #~ for ax in axs:
        #~ for name, freq_band in freqency_bands.items():
            #~ ax.axvline(freq_band[0])
            #~ ax.axvline(freq_band[1])
    #~ ax.set_xlim(0, 2)
    #~ plt.show()

    


if __name__ == '__main__':
    test_ecg()

