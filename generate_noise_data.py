from simulation import run_simulation
from multiprocessing import Pool


def f(params):
    outfolder = 'results/{}/'.format(params['key'])
    run_simulation(params, outfolder, solver=params['solver'])


noise_simgas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 1e1, 1e2]

parameters = [{
    'key': 'noise_to_square_right_inverse',
    'n_its': 500,
    'time': 'undefined',
    'positions': [20],
    'complexities': [5],
    'anchors': [8],
    'noise_sigmas': noise_simgas,
    'success_thresholds': [0.0] * len(noise_simgas),
    'noise_to_square': True,
    'solver': "alternativePseudoInverse"
}, {
    'key': 'noise_right_inverse_weighted',
    'n_its': 500,
    'time': 'undefined',
    'positions': [20],
    'complexities': [5],
    'anchors': [8],
    'noise_sigmas': noise_simgas,
    'success_thresholds': [0.0] * len(noise_simgas),
    "noise_to_square": False,
    'solver': 'weightedPseudoInverse'
}, {
    'key': 'noise_right_inverse',
    'n_its': 500,
    'time': 'undefined',
    'positions': [20],
    'complexities': [5],
    'anchors': [8],
    'noise_sigmas': noise_simgas,
    'success_thresholds': [0.0] * len(noise_simgas),
    "noise_to_square": False,
    'solver': 'alternativePseudoInverse'
}]

with Pool(3) as p:
    p.map(f, parameters)
