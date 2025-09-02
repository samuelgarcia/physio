import numpy as np
from pathlib import Path

from physio import compute_respiration, detect_respiration_cycles, preprocess

# read signals
example_folder = Path(__file__).parents[1] / 'examples'
raw_resp1 = np.load(example_folder / 'resp_airflow1.npy')
raw_resp3 = np.load(example_folder / 'resp_belt3.npy')
srate = 1000.



def test_compute_respiration():
    
    resp, resp_cycles = compute_respiration(raw_resp1, srate, parameter_preset='human_airflow')
    # print(resp.shape)
    # print(resp_cycles.shape)
    # print(resp_cycles)
    # print(resp_cycles.dtypes)



def test_detect_respiration_cycles_airflow():

    # airflow : low level
    resp = preprocess(raw_resp1, srate, band=25., btype='lowpass', ftype='bessel', order=5, normalize=False)
    cycles = detect_respiration_cycles(resp, srate, method="crossing_baseline", baseline_mode='median',
                              inspiration_adjust_on_derivative=False)
    # print(cycles.shape)


    # airflow : parameters
    resp, resp_cycles = compute_respiration(raw_resp1, srate, parameter_preset='human_airflow')
    # inspi_inds = resp_cycles['inspi_index'].values
    # expi_inds = resp_cycles['expi_index'].values
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(resp)
    # ax.scatter(inspi_inds, resp[inspi_inds], color='g')
    # ax.scatter(expi_inds, resp[expi_inds], color='r')
    # plt.show()



def test_detect_respiration_cycles_belt():

    # belt low level
    # resp = preprocess(raw_resp3, srate, band=25., btype='lowpass', ftype='bessel', order=5, normalize=False)
    # cycles = detect_respiration_cycles(resp, srate, method="min_max")
    # print(cycles.shape)
    # inspi_inds = cycles[:, 0]
    # expi_inds = cycles[:, 1]    

    # belt auto params
    params = dict(
        sensor_type='belt',
        preprocess=dict(band=15., btype='lowpass', ftype='bessel', order=5, normalize=False),
        # smooth=dict(win_shape='gaussian', sigma_ms=40.0),
        # baseline=None,
        smooth=None,
        cycle_detection=dict(method="min_max"),
        # cycle_clean=None,

        cycle_clean=[
            dict(variable_name="inflation_amplitude", low_limit_log_ratio=4.),
            dict(variable_name="deflation_amplitude", low_limit_log_ratio=4.),
        ]

    )
    resp, resp_cycles = compute_respiration(raw_resp3, srate, parameter_preset=None, parameters=params)
    print(resp_cycles.shape)
    inspi_inds = resp_cycles['inspi_index'].values
    expi_inds = resp_cycles['expi_index'].values
    print(resp_cycles[['inflation_amplitude', 'deflation_amplitude']])




    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(resp)
    ax.scatter(inspi_inds, resp[inspi_inds], color='g')
    ax.scatter(expi_inds, resp[expi_inds], color='r')
    plt.show()




if __name__ == '__main__':
    # test_compute_respiration()
    # test_detect_respiration_cycles_airflow()
    test_detect_respiration_cycles_belt()
