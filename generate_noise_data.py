from simulation import run_simulation
from multiprocessing import Pool


def f(params):
    outfolder = 'results/{}/'.format(params['key'])
    run_simulation(params, outfolder, solver=params['solver'], verbose=True)


noise_simgas = [1e-1, 2e-1, 5e-1, 1e0, 2e0, 5e0, 1e1]

parameters = [{
    'key': 'noise_to_square_right_inverse_2',
    'n_its': 10,
    'time': 'undefined',
    'positions': [200],
    'complexities': [3],
    'anchors': [6],
    'noise_sigmas': noise_simgas,
    'success_thresholds': [0.0] * len(noise_simgas),
    'noise_to_square': True,
    'solver': 'alternativePseudoInverse',
    'sampling_strategy': 'single_time'
}, {
    'key': 'noise_right_inverse_weighted_2',
    'n_its': 10,
    'time': 'undefined',
    'positions': [200],
    'complexities': [3],
    'anchors': [6],
    'noise_sigmas': noise_simgas,
    'success_thresholds': [0.0] * len(noise_simgas),
    "noise_to_square": False,
    'solver': 'weightedPseudoInverse',
    'sampling_strategy': 'single_time'
}, {
    'key': 'noise_right_inverse_2',
    'n_its': 10,
    'time': 'undefined',
    'positions': [200],
    'complexities': [3],
    'anchors': [6],
    'noise_sigmas': noise_simgas,
    'success_thresholds': [0.0] * len(noise_simgas),
    "noise_to_square": False,
    'solver': 'alternativePseudoInverse',
    'sampling_strategy': 'single_time'
}]

with Pool(3) as p:
    p.map(f, parameters)
