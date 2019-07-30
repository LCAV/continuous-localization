#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
draw_curve.py: draw trajectory and save coordinates. 
"""

import numpy as np
import matplotlib.pyplot as plt

from plotting_tools import make_dirs_safe


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
            print('x = {}, y = {}'.format(ix, iy))
            if (ix is not None) and (iy is not None):
                self.xs.append(ix)
                self.ys.append(iy)

            self.line.set_data(self.xs, self.ys)
            self.line.figure.canvas.draw()

    def onrelease(self, event):
        self.start = False

        if self.fname != '':
            self.fig.savefig(self.fname + '.png')
            coords = np.array([self.xs, self.ys])
            np.savetxt(self.fname + '.txt', coords, fmt='%.5f', delimiter=',')

            print('saved as {}.txt,png'.format(self.fname))

        [self.fig.canvas.mpl_disconnect(cid) for cid in self.cids]


if __name__ == "__main__":
    import time

    fig, ax = plt.subplots()
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])

    coords = np.loadtxt('fitting/test.txt', delimiter=',')
    ax.plot(coords[0], coords[1], label='previous', color='red')
    line, = ax.plot([], [], label='new')
    ax.legend()
    ax.set_aspect('equal')

    #fname = 'fitting/random_{:.0f}'.format(time.time())
    #fname = 'fitting/circle'
    fname = 'fitting/complicated'
    #fname = 'fitting/test'
    make_dirs_safe(fname)
    print('saving as', fname)

    ch = TrajectoryCreator(line, fname)
    ch.connect()
    print('figure was closed, and now what?')

    coords = np.loadtxt('{}.txt'.format(fname), delimiter=',')
    print(coords.shape)

    # fit trajectory times and coefficients to obtained trajectory.
