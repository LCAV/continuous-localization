#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trajectory.py: Contains the Trajectory class.

The units can be interpreted as 1m
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from global_variables import DIM, TMAX, TAU, ROBOT_WIDTH, EPSILON


class Trajectory(object):
    """ Trajectory class.

    :member dim: dimension (2 or 3)
    :member n_complexity: complexity of trajectory. 
    :member coeffs: coefficients of trajectory. (dim x n_complexity)
    :member model: trajectory model,
    either bandlimited, full_bandlimited (both sines and cosines) or polynomial.
    """

    def __init__(self, n_complexity=3, dim=DIM, model='bandlimited', tau=TAU, full_period=False):
        self.dim = dim
        self.n_complexity = n_complexity
        self.coeffs = None
        self.model = model
        if self.model == 'full_bandlimited':
            full_period = True
        self.params = {'tau': tau, 'full_period': full_period}
        self.set_coeffs()

    def copy(self):
        new = Trajectory(self.n_complexity, self.dim, self.model, self.params['tau'])
        new.set_coeffs(coeffs=self.coeffs)
        return new

    def get_times(self, n_samples):
        """ Get times appropriate for this trajectory model. """
        if self.model == 'polynomial':
            times = np.linspace(0, TMAX, n_samples)
        elif self.model == 'bandlimited' or self.model == 'full_bandlimited':
            part = 1.0 if self.params['full_period'] else 0.5
            times = np.linspace(0, part * self.params['tau'], n_samples)
        else:
            raise NotImplementedError(self.model)

        return times

    def get_basis(self, n_samples=None, times=None):
        """ Get basis vectors evaluated at specific times. 

        :param n_samples: number of samples. 
        :param times: vector of times of length n_samples

        :return: basis vector matrix (n_complexity x n_samples)
        """
        if n_samples is None and times is None:
            raise NotImplementedError('Need to give times or n_samples.')
        elif times is None:
            times = self.get_times(n_samples)
        elif times is not None and n_samples is not None:
            raise AttributeError('Cannot give n_samples and times.')
        elif times is not None and n_samples is None:
            n_samples = len(times)
        else:
            raise NotImplementedError('case not treated:', n_samples, times)

        k = np.reshape(range(self.n_complexity), [self.n_complexity, 1])
        n = np.reshape(times, [1, n_samples])
        if self.model == 'bandlimited':
            return np.cos(2 * np.pi * k * n / self.params['tau'])
        elif self.model == 'polynomial':
            return np.power(n, k)
        elif self.model == 'full_bandlimited':
            assert self.n_complexity % 2 == 1, \
                "full bandlimited model requires odd number of coefficients"
            k = np.reshape(range(math.ceil(self.n_complexity / 2)), [math.ceil(self.n_complexity / 2), 1])
            basis = np.ones((self.n_complexity, n_samples))
            basis[::2] = np.cos(2 * np.pi * k * n / self.params['tau'])
            basis[1::2] = np.sin(2 * np.pi * k[1:] * n / self.params['tau'])
            return basis
        else:
            raise NotImplementedError(self.model)

    def get_basis_prime(self, times=None):
        """ Get basis vector derivatives evaluated at specific times. 
        :param times: vector of times of length n_samples
        :return: 1st derivative (in time) of basis vector matrix (n_complexity x 
n_samples)
        """
        n_samples = len(times)
        n = np.reshape(times, [1, n_samples])
        if self.model == 'bandlimited':
            k = np.reshape(range(self.n_complexity), [self.n_complexity, 1])
            return -2 * np.pi * k / self.params['tau'] * np.sin(2 * np.pi * k * n / self.params['tau'])
        elif self.model == 'polynomial':
            k_reduced = np.reshape(range(self.n_complexity - 1), [self.n_complexity - 1, 1])
            return np.r_[np.zeros((1, n_samples)), (k_reduced + 1) * np.power(n, k_reduced)]
        elif self.model == 'full_bandlimited':
            assert self.n_complexity % 2 == 1, \
                "full bandlimited model requires odd number of coefficients"
            k = np.reshape(range(math.ceil(self.n_complexity / 2)), [math.ceil(self.n_complexity / 2), 1])
            basis = np.ones((self.n_complexity, n_samples))
            basis[::2] = -2 * np.pi * k / self.params['tau'] * np.sin(2 * np.pi * k * n / self.params['tau'])
            basis[1::2] = 2 * np.pi * k[1:] / self.params['tau'] * np.cos(2 * np.pi * k[1:] * n / self.params['tau'])
            return basis
        else:
            raise NotImplementedError(self.model)

    def get_basis_twoprime(self, times=None):
        """ Get basis vector second derivatives evaluated at specific times. 
        :param times: vector of times of length n_samples
        :return: 2nd derivative (in time) of basis vector matrix (n_complexity x 
n_samples)
        """
        n_samples = len(times)
        n = np.reshape(times, [1, n_samples])
        if self.model == 'bandlimited':
            k = np.reshape(range(self.n_complexity), [self.n_complexity, 1])
            return -(2 * np.pi * k / self.params['tau'])**2 * np.cos(2 * np.pi * k * n / self.params['tau'])
        elif self.model == 'polynomial':
            k_reduced = np.reshape(range(self.n_complexity - 2), [self.n_complexity - 2, 1])
            return np.r_[np.zeros((2, n_samples)), (k_reduced + 1) * (k_reduced + 2) * np.power(n, k_reduced)]
        elif self.model == 'full_bandlimited':
            assert self.n_complexity % 2 == 1, \
                "full bandlimited model requires odd number of coefficients"
            k = np.reshape(range(math.ceil(self.n_complexity / 2)), [math.ceil(self.n_complexity / 2), 1])
            basis = np.ones((self.n_complexity, n_samples))
            basis[::2] = -(2 * np.pi * k / self.params['tau'])**2 * np.cos(2 * np.pi * k * n / self.params['tau'])
            basis[1::2] = -(2 * np.pi * k[1:] / self.params['tau'])**2 * np.sin(
                2 * np.pi * k[1:] * n / self.params['tau'])
            return basis
        else:
            raise NotImplementedError(self.model)

    def set_coeffs(self, seed=None, coeffs=None, dimension=5):
        if seed is not None:
            np.random.seed(seed)

        if coeffs is None:
            self.coeffs = dimension * \
                np.random.rand(self.dim, self.n_complexity)
        else:
            self.coeffs = coeffs

        dim = self.coeffs.shape[0]
        self.Z_opt = np.vstack(
            [np.hstack([np.eye(dim), self.coeffs]),
             np.hstack([self.coeffs.T, self.coeffs.T @ self.coeffs])])

    def get_sampling_points(self, basis=None, seed=None):
        """ Get points where we get measurements.
        
        """
        points = self.coeffs @ basis
        return points

    def get_continuous_points(self):
        basis_cont = self.get_basis(n_samples=1000)
        trajectory_cont = self.get_sampling_points(basis=basis_cont)
        return trajectory_cont

    def plot(self, basis, mask=None, **kwargs):
        """ Plot continuous and sampled version.

        :param times: times of sampling points.
        :param mask: optional measurements mask (to plot missing measurements)
        :param kwargs: any additional kwargs passed to plt.scatter()

        """

        trajectory_cont = self.get_continuous_points()
        trajectory = self.get_sampling_points(basis=basis)

        if mask is not None:
            trajectory = trajectory[:, np.any(mask[:, :] != 0, axis=1)]

        cont_kwargs = {k: val for k, val in kwargs.items() if k != 'marker'}
        plt.plot(*trajectory_cont, **cont_kwargs)
        # avoid having two labels of same thing.
        pop_labels = ['label', 'linestyle']
        for pop_label in pop_labels:
            if pop_label in kwargs.keys():
                kwargs.pop(pop_label)
        plt.scatter(*trajectory, **kwargs)

    def plot_connections(self, basis, anchors, mask, **kwargs):
        trajectory = self.get_sampling_points(basis=basis)
        ns, ms = np.where(mask)
        for n, m in zip(ns, ms):
            p1 = trajectory[:, n]
            p2 = anchors[:, m]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], **kwargs)

    def plot_number_measurements(self, basis, mask=None, legend=False):
        #  mask is n_samples x n_anchors.
        trajectory = self.get_sampling_points(basis=basis)
        if legend:
            label1 = '1'
            label2 = '2'
            label3 = '>2'
        else:
            label1 = label2 = label3 = None
        for i in range(trajectory.shape[1]):
            point = trajectory[:, i]
            if np.sum(mask[i, :]) == 1:
                plt.scatter(*point, color='orange', label=label1)
                label1 = None
            elif np.sum(mask[i, :]) == 2:
                plt.scatter(*point, color='red', label=label2)
                label2 = None
            elif np.sum(mask[i, :]) > 2:
                plt.scatter(*point, color='green', label=label3)
                label3 = None
        if legend:
            plt.legend(title='# measurements')

    def scale_bounding_box(self, box_dims, keep_aspect_ratio=False):
        """Scale trajectory to a given size.
        
        :param box_dims: the dimensions of the desired bounding box, 
        the bounding box is assumed to begin at (0, 0)
        :param keep_aspect_ratio: if true, the second dimension of the bounding 
        box is ignored, and coefficients are scaled the same in both dimensions
        
        :return: true bounding box dimensions,
        no mater if aspect ratio was preserved or not
        """

        points = self.get_continuous_points()
        shift = np.min(points, axis=1)
        points = points - shift[:, None]
        self.coeffs[:, 0] -= shift
        scale = box_dims / np.max(points, axis=1)
        if keep_aspect_ratio:
            self.coeffs = self.coeffs * scale[0]
            box_dims[1] = np.max(points[1, :]) * scale[0]
        else:
            self.coeffs = self.coeffs * scale[:, None]
        return box_dims

    def get_times_uniform_in_path(self, n_samples=None, step_distance=None, time_steps=10000, plot=False):
        """Calculate numerically times equivalent to uniform sampling in
        distance travelled.
        
        It calculates the cumulative integral over small steps, and picks
        as a sample time the first time after the integral reaches expected
        distance at this step."""
        times = self.get_times(n_samples=time_steps)
        basis_prime = self.get_basis_prime(times=times)
        velocities = self.coeffs.dot(basis_prime)

        time_differences = times[1:] - times[:-1]
        speeds = np.linalg.norm(velocities, axis=0)
        distances = np.cumsum((speeds[1:] + speeds[:-1]) / 2 * time_differences)

        if n_samples is not None:
            uniform_distances = np.arange(n_samples) * distances[-1] / (n_samples - 1)
        elif step_distance is not None:
            uniform_distances = np.arange(distances[-1], step=step_distance)
        else:
            raise ValueError("Either n_samples or step_distance has to be provided")

        uniform_path_times = []
        errors = []
        i = 0
        for next_distance in uniform_distances:
            while i < len(distances) - 1 and (distances[i] < next_distance):
                i = i + 1
            errors.append(distances[i] - next_distance)
            uniform_path_times.append(times[i])

        if plot:
            plt.figure()
            plt.plot(times[1:], distances, label="uniform in time")
            plt.plot(uniform_path_times, uniform_distances, "-*", label="uniorm in distance")
            plt.title("distance traveled")
            plt.legend()
            plt.show()

        return np.array(uniform_path_times), uniform_distances, np.array(errors)

    def get_local_frame(self, times):
        """Calculate the local frame and the speeds.

        The frame is defined like Darboux Frame,
        https://en.wikipedia.org/wiki/Darboux_frame,
        except we do not need the third "tangent normal" vector in 2D.

        The speed is returned because it is useful
        """

        # compute derivatives of the basis vectors.
        basis_prime = self.get_basis_prime(times=times)
        velocities = self.coeffs.dot(basis_prime)
        # avoid division by zero for small speeds
        speeds = np.linalg.norm(velocities, axis=0) + EPSILON
        tangents = velocities / speeds
        normals = np.empty_like(tangents)

        # Using the fact that we are in 2D and normal vectors are perpendicular
        # to tangent vectors
        normals[0, :] = -tangents[1, :]
        normals[1, :] = tangents[0, :]
        return tangents, normals, speeds

    def get_curvatures(self, times, ax=None):
        tangents, normal_vectors, speeds = self.get_local_frame(times)

        basis_twoprime = self.get_basis_twoprime(times=times)
        accelerations = self.coeffs.dot(basis_twoprime)

        # https://en.wikipedia.org/wiki/Curvature
        # Section: Local expressions
        # We also use the fact that tangent * speed = velocity
        curvature_values = np.sum(normal_vectors * accelerations, axis=0) / (speeds**2)
        curvatures = curvature_values * normal_vectors

        radii = 1.0 / curvature_values
        radius_vectors = radii * normal_vectors

        # plot
        if ax is not None:
            basis = self.get_basis(times=times)
            sample_points = self.get_sampling_points(basis=basis)
            for i in range(0, sample_points.shape[1]):
                s = sample_points[:, i]
                t = s + tangents[:, i]
                n = s + normal_vectors[:, i]
                r = s + radius_vectors[:, i]
                plt.plot([s[0], t[0]], [s[1], t[1]], ':', color='black')
                plt.plot([s[0], n[0]], [s[1], n[1]], ':', color='orange')
                plt.plot([s[0], r[0]], [s[1], r[1]], ':', color='blue')

                radius = radii[i]
                center = r
                circ = Circle(xy=center, radius=radius, alpha=0.1, color='blue')
                plt.scatter(*center, color='blue', marker='+')
                plt.gca().add_artist(circ)

        return radii, tangents, curvatures

    def get_left_and_right_points(self, times, width=ROBOT_WIDTH, ax=None):

        basis = self.get_basis(times=times)
        sample_points = self.get_sampling_points(basis=basis)
        tangents, normal_vectors, _ = self.get_local_frame(times)

        points_left = sample_points - width / 2. * normal_vectors
        points_right = sample_points + width / 2. * normal_vectors

        label = 'left and right wheel'
        if ax is not None:
            for cl, cr in zip(points_left.T, points_right.T):
                if all(np.isnan(cl)):
                    continue
                plt.scatter(*cl, color='red', marker='+', label=label)
                label = None
                plt.scatter(*cr, color='red', marker='+', label=label)

        return points_left, points_right
