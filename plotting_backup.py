"""
plotting_backup: This module exists only to temporarily
host hold plotting functions. They should be replaced by their more modular versions soon!!
"""
import matplotlib.pyplot as plt
import numpy as np

from other_algorithms import pointwise_srls
from plotting_tools import remove_ticks, add_scalebar
from solvers import trajectory_recovery


#TODO(FD) below too functions have too much computation. Try to split
# plotting and processing more cleanly.
def plot_subsample_old(traj, D, times, anchors, full_df, n_measurements_list):
    import hypothesis as h

    basis = traj.get_basis(times=times)
    fig, axs = plt.subplots(1, len(n_measurements_list), sharex=True, sharey=True)

    alpha = 1.0
    num_seeds = 3
    for ax, n_measurements in zip(axs, n_measurements_list):
        label = 'ours'

        coeffs = np.empty([traj.dim, traj.n_complexity, 0])
        colors = {0: 0, 1: 2, 2: 3}
        for seed in range(num_seeds):
            np.random.seed(seed)
            indices = np.random.choice(D.shape[0], n_measurements, replace=False)

            D_small = D[indices, :]
            mask = (D_small > 0).astype(np.float)

            p = np.sort(np.sum(mask, axis=0))[::-1]
            if not h.limit_condition(list(p), traj.dim + 1, traj.n_complexity):
                print("insufficient rank")

            times_small = np.array(times)[indices]
            basis_small = traj.get_basis(times=times_small)

            Chat = trajectory_recovery(D_small, anchors[:2, :], basis_small, weighted=True)

            coeffs = np.dstack([coeffs, Chat])
            traj.set_coeffs(coeffs=Chat)

            traj.plot_pretty(times=times, color='C{}'.format(colors[seed]), ax=ax, alpha=alpha)

        Chat_avg = np.mean(coeffs, axis=2)
        traj.set_coeffs(coeffs=Chat_avg)
        traj.plot_pretty(times=times, color='C0', label=label, ax=ax)

        points, used_indices = pointwise_srls(D, anchors, traj, indices)
        label = 'SRLS'
        for x in points:
            ax.scatter(*x, color='C1', label=label, s=4.0)
            label = None

        ax.plot(full_df.px, full_df.py, ls=':', linewidth=1., color='black', label='GPS')
        remove_ticks(ax)
        ax.set_title('N = {}'.format(n_measurements), y=-0.22)
    add_scalebar(axs[0], 20, loc='lower left')
    return fig, axs


def plot_complexities_old(traj, D, times, anchors, full_df, list_complexities, srls=True):

    fig, axs = plt.subplots(1, len(list_complexities), sharex=True, sharey=True)
    for ax, n_complexity in zip(axs, list_complexities):
        traj.set_n_complexity(n_complexity)
        basis = traj.get_basis(times=times)

        Chat = trajectory_recovery(D, anchors[:2, :], basis, weighted=True)

        traj.set_coeffs(coeffs=Chat)

        traj.plot_pretty(times=times, color="C0", label='ours', ax=ax)
        ax.plot(full_df.px, full_df.py, color='black', ls=':', linewidth=1., label='GPS')
        ax.set_title('K = {}'.format(traj.n_complexity))

        remove_ticks(ax)
        if srls:
            indices = range(D.shape[0])[::3]
            points, used_indices = pointwise_srls(D, anchors, traj, indices)
            label = 'SRLS'
            for x in points:
                ax.scatter(*x, color='C1', label=label, s=4.0)
                label = None

    add_scalebar(axs[0], 20, loc='lower left')
    return fig, axs
