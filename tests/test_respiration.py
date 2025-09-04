import numpy as np
from pathlib import Path

from physio import compute_respiration, detect_respiration_cycles, preprocess

# read signals
example_folder = Path(__file__).parents[1] / 'examples'
raw_resp1 = np.load(example_folder / 'resp_airflow1.npy')
raw_resp2 = np.load(example_folder / 'resp_airflow2.npy')
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


    # airflow : parameter preset
    resp, resp_cycles = compute_respiration(raw_resp1, srate, parameter_preset='human_airflow')
    # print(resp_cycles.shape)
    resp, resp_cycles = compute_respiration(raw_resp2, srate, parameter_preset='human_airflow')

    # airflow : parameters manual
    parameters = dict(
        sensor_type='airflow',
        preprocess=dict(band=7., btype='lowpass', ftype='bessel', order=5, normalize=False),
        smooth=dict(win_shape='gaussian', sigma_ms=60.0),
        cycle_detection=dict(method="crossing_baseline", epsilon_factor1=20, epsilon_factor2=3., inspiration_adjust_on_derivative=False),
        baseline=dict(baseline_mode='median'),
        cycle_clean=dict(variable_names=['inspi_volume', 'expi_volume'], low_limit_log_ratio=4.5),
    )
    resp, resp_cycles = compute_respiration(raw_resp1, srate, parameter_preset=None, parameters=parameters)

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
    resp = preprocess(raw_resp3, srate, band=25., btype='lowpass', ftype='bessel', order=5, normalize=False)
    cycles = detect_respiration_cycles(resp, srate, method="min_max")
    
    # print(cycles.shape)
    # inspi_inds = cycles[:, 0]
    # expi_inds = cycles[:, 1]    

    # belt preset
    resp, resp_cycles = compute_respiration(raw_resp3, srate, parameter_preset='human_belt')


    # belt  params
    params = dict(
        sensor_type='belt',
        preprocess=dict(band=5., btype='lowpass', ftype='bessel', order=5, normalize=False),
        smooth=dict(win_shape='gaussian', sigma_ms=40.0),
        cycle_detection=dict(method="min_max", exclude_sweep_ms=200.),
        # cycle_clean=None,
        cycle_clean=dict(variable_names=["inspi_amplitude", "expi_amplitude"], low_limit_log_ratio=8.),
        # cycle_clean=dict(variable_names=["inspi_amplitude",], low_limit_log_ratio=8.),
        # cycle_clean=dict(variable_names=["expi_amplitude",], low_limit_log_ratio=8.),
    )
    resp, resp_cycles = compute_respiration(raw_resp3, srate, parameter_preset=None, parameters=params)
    
    # print(resp_cycles.shape)
    # print(resp_cycles[['inspi_amplitude', 'expi_amplitude']])



    inspi_inds = resp_cycles['inspi_index'].values
    expi_inds = resp_cycles['expi_index'].values
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(raw_resp3)
    ax.plot(resp)
    ax.scatter(inspi_inds, resp[inspi_inds], color='g')
    ax.scatter(expi_inds, resp[expi_inds], color='r')
    plt.show()




if __name__ == '__main__':
    # test_compute_respiration()
    # test_detect_respiration_cycles_airflow()
    test_detect_respiration_cycles_belt()
