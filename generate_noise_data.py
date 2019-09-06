from simulation import run_simulation

from multiprocessing import Pool


def f(params):
    outfolder = 'results/{}/'.format(params['key'])
    run_simulation(params, outfolder, solver=params['solver'], verbose=True)


noise_simgas = [1e-1, 1e0, 1e1]

parameters = [
    {
        'key': 'noise_to_square_right_inverse',
        'n_its': 1000,
        'time': 'undefined',
        'positions': [500],
        'complexities': [5],
        'anchors': [4],
        'noise_sigmas': noise_simgas,
        'success_thresholds': [0.0] * len(noise_simgas),
        'noise_to_square': True,
        'solver': 'alternativePseudoInverse',
        'sampling_strategy': 'single_time'
    },
    {
        'key': 'noise_right_inverse_weighted',
        'n_its': 1000,
        'time': 'undefined',
        'positions': [500],
        'complexities': [5],
        'anchors': [4],
        'noise_sigmas': noise_simgas,
        'success_thresholds': [0.0] * len(noise_simgas),
        "noise_to_square": False,
        'solver': 'weightedPseudoInverse',
        'sampling_strategy': 'single_time'
    },
    {
        'key': 'noise_right_inverse',
        'n_its': 1000,
        'time': 'undefined',
        'positions': [500],
        'complexities': [5],
        'anchors': [4],
        'noise_sigmas': noise_simgas,
        'success_thresholds': [0.0] * len(noise_simgas),
        "noise_to_square": False,
        'solver': 'alternativePseudoInverse',
        'sampling_strategy': 'single_time'
    },
    {
        'key': 'noise_and_anchors',
        'n_its': 1000,
        'time': 'undefined',
        'positions': [500],
        'complexities': [3],
        'anchors': [3, 5, 10],
        'noise_sigmas': [0.1],
        'success_thresholds': [0.0],
        'noise_to_square': True,
        'solver': 'alternativePseudoInverse',
        'sampling_strategy': 'single_time'
    },
]

with Pool(2) as p:
    p.map(f, parameters)
