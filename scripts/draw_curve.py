#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
draw_curve.py: draw trajectory and save coordinates. 
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
from os.path import abspath, dirname

sys.path.append(dirname(abspath(__file__)) + '/../source')


class TrajectoryCreator:
    def __init__(self, line, fname=''):
        self.start = False
        self.xs = []
        self.ys = []

        self.cids = []
        self.line = line
        self.fig = line.figure

        self.fname = fname

    def connect(self):
        cid_press = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.onmotion)
        cid_release = self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
        cid_release = self.fig.canvas.mpl_connect('close_event', self.onclose)
        self.cids = [cid_press, cid_motion, cid_release]
        plt.show()

    def onclick(self, event):
        if not self.start:
            ix, iy = event.xdata, event.ydata
            self.xs.append(ix)
            self.ys.append(iy)
            self.start = True

    def onmotion(self, event):
        if self.start:
            ix, iy = event.xdata, event.ydata
            if (ix is not None) and (iy is not None):
                self.xs.append(ix)
                self.ys.append(iy)

            self.line.set_data(self.xs, self.ys)
            self.line.figure.canvas.draw()

    def onrelease(self, event):
        self.start = False

    def onclose(self, event):
        self.start = False

        coords = np.array([self.xs, self.ys])
        if (self.fname != '') and (len(self.xs) > 0):
            self.fig.savefig(self.fname + '.png')
            np.savetxt(self.fname + '.txt', coords, fmt='%.5f', delimiter=',')

            print('Saved as {}.txt and *.png'.format(self.fname))
        elif len(self.xs) == 0:
            print('Warning: empty trajectory, did not save.')

        [self.fig.canvas.mpl_disconnect(cid) for cid in self.cids]


if __name__ == "__main__":
    from plotting_tools import make_dirs_safe
    from coordinate_fitting import fit_trajectory_and_times
    from trajectory import Trajectory

    fname = '../results/fitting/plaza'

    # draw a trajectory
    fig, ax = plt.subplots()
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])

    line, = ax.plot([], [], label='new')
    ax.legend()
    ax.set_aspect('equal')

    make_dirs_safe(fname)
    print('Saving as {} as soon as figure is closed.'.format(fname))

    ch = TrajectoryCreator(line, fname)
    ch.connect()

    # fit the trajectory
    coords_original = np.loadtxt(fname + '.txt', delimiter=',')
    n_complexity = 11
    dim = 2
    model = 'full_bandlimited'
    trajectory = Trajectory(n_complexity=n_complexity, dim=dim, model=model, full_period=True)
    coeffs, times = fit_trajectory_and_times(coords_original, trajectory, max_iter=10)
    trajectory.set_coeffs(coeffs=coeffs)

    # plot results
    basis = trajectory.get_basis(times=times)
    coords_reconstructed = trajectory.get_sampling_points(basis=basis)
    plt.figure()
    plt.title('Trajectory fitting result')
    plt.plot(*coords_original, color='green', label='original')
    plt.plot(*coords_reconstructed, color='green', linestyle=':', label='reconstructed')
    plt.legend()
    plt.savefig(fname + '_fit.png')
    print('Saved as', fname + '_fit.png')
    plt.show()
