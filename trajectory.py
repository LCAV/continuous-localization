#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trajectory.py: Contains the Trajectory class.

The units can be interpreted as 1m.

"""

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from global_variables import DIM, TMAX, TAU, ROBOT_WIDTH, EPSILON

import copy


class Trajectory(object):
    """ Trajectory class.

    :member dim: dimension (2 or 3)
    :member n_complexity: complexity of trajectory
    :member coeffs: coefficients of trajectory (dim x n_complexity)
    :member model: trajectory model either bandlimited, full_bandlimited (both sines and cosines) or polynomial.

    """
    def __init__(self,
                 n_complexity=3,
                 dim=DIM,
                 model='bandlimited',
                 period=TAU,
                 full_period=False,
                 seed=None,
                 coeffs=None,
                 name=None):
        """

        :param n_complexity: complexity of trajectory.
        :param dim: dimension of the trajectory.
        :param model: trajectory model type (either bandlimited, full_bandlimited (both sines and cosines) or
        polynomial).
        :param period: period in second for bandlimited trajectories.
        :param full_period: if true, the default time of trajectory is 0 to period, else it is only a 0 to 0.5 of period
        :param seed: random seed to generate coefficients
        :param coeffs: array of coefficients of shape (dim x n_complexity). If it is given, the dimensions and
        complexity are inferred form it.
        :param name: name of trajectory, for plotting

        """
        if coeffs is not None:
            dim, n_complexity = coeffs.shape
        self.dim = dim
        self.n_complexity = n_complexity
        self.coeffs = None
        self.model = model
        if self.model == 'full_bandlimited':
            full_period = True
        self.period = period
        self.params = {'full_period': full_period}
        if name is not None:
            self.params["name"] = name
        self.set_coeffs(seed=seed, coeffs=coeffs)

    def copy(self):
        new = Trajectory(self.n_complexity, self.dim, self.model, self.period, coeffs=np.copy(self.coeffs))
        new.params = copy.deepcopy(self.params)
        return new

    def get_times(self, n_samples):
        """ Get times appropriate for this trajectory model. """
        if self.model == 'polynomial':
            times = np.linspace(0, self.period, n_samples)
        elif self.model == 'bandlimited' or self.model == 'full_bandlimited':
            part = 1.0 if self.params['full_period'] else 0.5
            times = np.linspace(0, part * self.period, n_samples)
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
            basis = np.ones((self.n_complexity, n_samples))
            basis[1:] = 2 * np.cos(2 * np.pi * k[1:] * n / self.period)
            return basis
        elif self.model == 'polynomial':
            return np.power(n, k)
        elif self.model == 'full_bandlimited':
            assert self.n_complexity % 2 == 1, \
                "full bandlimited model requires odd number of coefficients"
            k = np.reshape(range(math.ceil(self.n_complexity / 2)), [math.ceil(self.n_complexity / 2), 1])
            basis = np.ones((self.n_complexity, n_samples))
            basis[2::2] = 2 * np.cos(2 * np.pi * k[1:] * n / self.period)
            basis[1::2] = 2 * np.sin(2 * np.pi * k[1:] * n / self.period)
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
            return -4 * np.pi * k / self.period * np.sin(2 * np.pi * k * n / self.period)
        elif self.model == 'polynomial':
            k_reduced = np.reshape(range(self.n_complexity - 1), [self.n_complexity - 1, 1])
            return np.r_[np.zeros((1, n_samples)), (k_reduced + 1) * np.power(n, k_reduced)]
        elif self.model == 'full_bandlimited':
            assert self.n_complexity % 2 == 1, \
                "full bandlimited model requires odd number of coefficients"
            k = np.reshape(range(math.ceil(self.n_complexity / 2)), [math.ceil(self.n_complexity / 2), 1])
            basis = np.ones((self.n_complexity, n_samples))
            basis[::2] = -4 * np.pi * k / self.period * np.sin(2 * np.pi * k * n / self.period)
            basis[1::2] = 4 * np.pi * k[1:] / self.period * np.cos(2 * np.pi * k[1:] * n / self.period)
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
            return -2 * (2 * np.pi * k / self.period)**2 * np.cos(2 * np.pi * k * n / self.period)
        elif self.model == 'polynomial':
            k_reduced = np.reshape(range(self.n_complexity - 2), [self.n_complexity - 2, 1])
            return np.r_[np.zeros((2, n_samples)), (k_reduced + 1) * (k_reduced + 2) * np.power(n, k_reduced)]
        elif self.model == 'full_bandlimited':
            assert self.n_complexity % 2 == 1, \
                "full bandlimited model requires odd number of coefficients"
            k = np.reshape(range(math.ceil(self.n_complexity / 2)), [math.ceil(self.n_complexity / 2), 1])
            basis = np.ones((self.n_complexity, n_samples))
            basis[::2] = -2 * (2 * np.pi * k / self.period)**2 * np.cos(2 * np.pi * k * n / self.period)
            basis[1::2] = -2 * (2 * np.pi * k[1:] / self.period)**2 * np.sin(2 * np.pi * k[1:] * n / self.period)
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
            if coeffs.shape[1] != self.n_complexity:
                print('Warning: coeffs mismatch', coeffs.shape[1], self.n_complexity)
            self.coeffs = coeffs

        dim = self.coeffs.shape[0]
        self.Z_opt = np.vstack(
            [np.hstack([np.eye(dim), self.coeffs]),
             np.hstack([self.coeffs.T, self.coeffs.T @ self.coeffs])])

    def set_n_complexity(self, n_complexity):
        """ Set new complexity and cut or pad coefficients with zeros if necessary. """
        new_coeffs = np.zeros((self.dim, n_complexity))
        keep = min(self.n_complexity, n_complexity)
        new_coeffs[:, :keep] = self.coeffs[:self.dim, :keep]
        self.coeffs = new_coeffs
        self.n_complexity = n_complexity

    def get_sampling_points(self, times=None, basis=None):
        """ Get points where we get measurements.

        :return: sampling points of shape dimxN.
        
        """
        if basis is None:
            basis = self.get_basis(times=times)
        points = self.coeffs @ basis
        return points

    def get_continuous_points(self, times=None):
        if times is None:
            basis_cont = self.get_basis(n_samples=1000)
        else:
            times_cont = np.linspace(times[0], times[-1], 1000)
            basis_cont = self.get_basis(times=times_cont)
        trajectory_cont = self.get_sampling_points(basis=basis_cont)
        return trajectory_cont

    def plot(self, basis=None, mask=None, times=None, **kwargs):
        """ Plot continuous and sampled version.

        :param basis: basis of sampling points. Only plot continuous version if not given.
        :param mask: optional measurements mask (to plot missing measurements)
        :param kwargs: any additional kwargs passed to plt.scatter()

        """
        if basis is not None:
            print(
                'Warning: it is now preferable to pass times instead of basis to the plotting function.  The basis argument is ignored when plotting the continuous trajectory.'
            )

        trajectory_cont = self.get_continuous_points(times=times)

        if times is not None:
            if basis is not None:
                print('Warning: overwriting basis with times.')
            basis = self.get_basis(times=times)

        if "ax" in kwargs:
            ax = kwargs["ax"]
            kwargs.pop("ax")
        else:
            fig, ax = plt.subplots()

        cont_kwargs = {k: val for k, val in kwargs.items() if (k != 'marker')}
        ax.plot(*trajectory_cont[:2], **cont_kwargs)

        if "name" in self.params:
            ax.set_title(self.params["name"])

        if basis is not None:
            trajectory = self.get_sampling_points(basis=basis)

            if mask is not None:
                trajectory = trajectory[:, np.any(mask != 0, axis=1)]

            # avoid having two labels of same thing.
            pop_labels = ['label', 'linestyle']
            for pop_label in pop_labels:
                if pop_label in kwargs.keys():
                    kwargs.pop(pop_label)
            ax.scatter(*trajectory[:2], **kwargs)
        return ax

    def plot_pretty(self, times=None, **kwargs):
        trajectory_cont = self.get_continuous_points(times=times)
        if "ax" in kwargs:
            ax = kwargs["ax"]
            kwargs.pop("ax")
        else:
            fig, ax = plt.subplots()
        cont_kwargs = {k: val for k, val in kwargs.items() if (k != 'marker')}
        ax.plot(*trajectory_cont[:2], **cont_kwargs)
        return ax

    def plot_connections(self, basis, anchors, mask, ax=None, **kwargs):
        trajectory = self.get_sampling_points(basis=basis)
        ns, ms = np.where(mask)
        for n, m in zip(ns, ms):
            p1 = trajectory[:, n]
            p2 = anchors[:, m]
            if ax is None:
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], **kwargs)
            else:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **kwargs)

    def plot_noisy_connections(self, basis, anchors, mask, D_noisy, ax=None, **kwargs):
        """ Plot measurements between trajectory points and anchors 
        with the noisy distances as connection lengths. 
        """
        trajectory = self.get_sampling_points(basis=basis)
        ns, ms = np.where(mask)
        for n, m in zip(ns, ms):
            d = np.sqrt(D_noisy[n, m])
            p1 = trajectory[:, n]
            p2 = anchors[:, m]
            v = p1 - p2
            alpha = np.arctan2(v[1], v[0])
            p3 = p2 + d * np.array((np.cos(alpha), np.sin(alpha)))
            if ax is None:
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], marker='o', **kwargs)
                plt.plot([p3[0], p2[0]], [p3[1], p2[1]], marker='o', **kwargs)
            else:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], marker='o', **kwargs)
                ax.plot([p3[0], p2[0]], [p3[1], p2[1]], marker='o', **kwargs)

    def plot_number_measurements(self, basis, mask=None, legend=False, ax=None):
        #  mask is n_samples x n_anchors.
        trajectory = self.get_sampling_points(basis=basis)
        if ax is None:
            ax = plt.gca()

        if legend:
            label1 = '1'
            label2 = '2'
            label3 = '>2'
        else:
            label1 = label2 = label3 = None
        for i in range(trajectory.shape[1]):
            point = trajectory[:, i]
            if np.sum(mask[i, :]) == 1:
                ax.scatter(*point, color='orange', label=label1)
                label1 = None
            elif np.sum(mask[i, :]) == 2:
                ax.scatter(*point, color='red', label=label2)
                label2 = None
            elif np.sum(mask[i, :]) > 2:
                ax.scatter(*point, color='green', label=label3)
                label3 = None
        if legend:
            ax.legend(title='# measurements')

    def scale_bounding_box(self, box_dims, keep_aspect_ratio=False):
        """Scale trajectory to a given size.
        
        :param box_dims: the dimensions of the desired bounding box (x, y), the bounding box is assumed to begin at (0, 0)
        :param keep_aspect_ratio: if true, the second dimension of the bounding box is ignored, and coefficients are scaled the same in both dimensions
        
        :return: true bounding box dimensions, no mater if aspect ratio was preserved or not

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

    def center(self):
        """Center trajectory so that the center of mass is at (0,0)"""
        points = self.get_continuous_points()
        self.coeffs[:, 0] -= np.mean(points, axis=1)

    def get_times_from_distances(self,
                                 n_samples=None,
                                 step_distance=None,
                                 time_steps=10000,
                                 plot=False,
                                 arbitrary_distances=None):
        """Calculate numerically times equivalent to given distances travelled.
        
        It calculates the cumulative integral over small steps, and picks
        as a sample time the first time after the integral reaches expected
        distance at this step.

        :param arbitrary_distances: if provided, returns times for those distances
        :param n_samples: if provided, n_samples distances are generated uniformly along the trajectory
        :param plot: if true, plot the distance against the (in model) time
        :param time_steps: number of steps for numerical integration
        :param step_distance: if provided, samples are generated step_distance apart through trajectory

        :return:
            triple:
            times at which samples have to be taken from model trajectory,
            distances at travelled in those times
            and approximation errors
        """
        times = self.get_times(n_samples=time_steps)
        basis_prime = self.get_basis_prime(times=times)
        velocities = self.coeffs.dot(basis_prime)

        time_differences = times[1:] - times[:-1]
        speeds = np.linalg.norm(velocities, axis=0)
        # generate distances only in the "natural" time of the trajectory
        # longer distances are generated on the fly
        cumulative_distances = np.concatenate([[0], np.cumsum((speeds[1:] + speeds[:-1]) / 2 * time_differences)])

        if arbitrary_distances is not None:
            if any(d < 0 for d in arbitrary_distances):
                raise ValueError("Provided distances have to be positive")
            distances = arbitrary_distances
        elif n_samples is not None:
            distances = np.arange(n_samples) * cumulative_distances[-1] / (n_samples - 1)
        elif step_distance is not None:
            distances = np.arange(cumulative_distances[-1], step=step_distance)
        else:
            raise ValueError("Either n_samples or step_distance or arbitrary distances has to be provided")

        new_times = []
        errors = []
        i = 0
        extra_distance = 0
        extra_time = 0

        for next_distance in distances:
            while cumulative_distances[i] < next_distance:
                i += 1
                # if we run out of precomputed distances, generate new ones
                # this requires basis at new times (starting where the previous times ended)
                # and the new distances have to be added on top of the distance traveled so far
                if i == len(cumulative_distances):
                    extra_time += times[-1]
                    extra_distance += cumulative_distances[-1]
                    basis_prime = self.get_basis_prime(times=times + extra_time)
                    velocities = self.coeffs.dot(basis_prime)
                    speeds = np.linalg.norm(velocities, axis=0)
                    cumulative_distances = np.cumsum((speeds[1:] + speeds[:-1]) / 2 * time_differences) + extra_distance
                    i = 0
            errors.append(cumulative_distances[i] - next_distance)
            new_times.append(extra_time + times[i])

        if plot:
            plt.figure()
            plt.plot(times, cumulative_distances, label="smooth")
            plt.plot(new_times, distances, "*", label="requested distances")
            plt.xlabel("time")
            plt.title("distance traveled")
            plt.legend()
            plt.show()

        assert len(new_times) == len(distances)

        return np.array(new_times), distances, np.array(errors)

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

        radii = 1.0 / (curvature_values + EPSILON)
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

    def get_left_and_right_arcs(self,
                                width=ROBOT_WIDTH,
                                time_steps=100000,
                                curvature_decimals=1,
                                min_max_distance=0.05,
                                plot=False):
        """Get the distances for left and right wheel sampled at positions suitable for the robot.

        If you need general method you may want to use get_left_and_right_points.

        :param width: width of the robot (in meters) that defines the positions of the wheels
        :param time_steps: to how many time steps discretize the trajectory during the calculation
        :param curvature_decimals: to how many decimal places to round the curvature. Curvature is used to decide where to give the robot new coordinates, and is not used directly in calculation of the paths. We are rounding the curvature and not the radius, because a small change for a small radius will lead to significant change in the trajectory. Rounding curvature at one decimal place corresponds to allowing for the biggest radius of 10m.
        :param min_max_distance: minimum distance (in meters) that at least one of the wheels have to travel in each segment
        :param plot: if true a plot is produced

        :return: two arrays of corresponding left and right wheel distances
        """
        times = self.get_times(n_samples=time_steps)
        points_left, points_right = self.get_left_and_right_points(times=times, width=width)

        ds_left = np.linalg.norm(points_left[:, 1:] - points_left[:, :-1], axis=0)
        ds_right = np.linalg.norm(points_right[:, 1:] - points_right[:, :-1], axis=0)
        curvature = np.round((ds_right - ds_left) / (ds_right + ds_left) / width, decimals=curvature_decimals)

        previous_c = curvature[0]
        current_left = ds_left[0]
        current_right = ds_right[0]
        distances_left = []
        distances_right = []
        new_times = []
        # Merge intervals that have similar curvature
        for idx, c in enumerate(curvature[1:]):
            # Merge intervals that are to short
            if c == previous_c or max(current_left, current_right) < min_max_distance:
                current_left += ds_left[idx]
                current_right += ds_right[idx]
            else:
                previous_c = c
                distances_left.append(current_left)
                distances_right.append(current_right)
                new_times.append(times[idx])
                current_left = ds_left[idx]
                current_right = ds_right[idx]
        distances_right.append(current_right)
        distances_left.append(current_left)
        new_times.append(times[len(curvature) - 1])

        distances_left = np.array(distances_left)
        distances_right = np.array(distances_right)
        new_times = np.array(new_times)

        if plot:
            plt.figure()
            plt.plot(np.cumsum(ds_left), np.cumsum(ds_right), label="continous")
            plt.scatter(np.cumsum(distances_left),
                        np.cumsum(distances_right),
                        label="discretized ({})".format(len(distances_left)),
                        marker='x',
                        color='C1')
            plt.legend()
            plt.show()
            print("minimum distance traveled by center: {:.4f}m".format(np.min((distances_left + distances_right) / 2)))
            print("minimum max distance: {:.4f}m".format(
                np.min([max(l, r) for l, r in zip(distances_left, distances_right)])))

        return distances_left, distances_right, new_times

    def get_name(self):
        """Gets name of the trajectory to for example display on plots"""
        if 'name' in self.params:
            return self.params['name']
        return self.model
