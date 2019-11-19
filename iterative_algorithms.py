#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iterative_algorithms.py: Contains functions to build up the trajectory iteratively over time. 

"""

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from solvers import trajectory_recovery


def verify_dimensions(D, anchors, basis, times):
    N, M = D.shape
    dim = anchors.shape[0]
    if anchors.shape[1] != M:
        raise ValueError(D.shape, anchors.shape, basis.shape, len(times))
    if basis.shape[1] != N:
        raise ValueError(D.shape, anchors.shape, basis.shape, len(times))
    if len(times) != N:
        raise ValueError(D.shape, anchors.shape, basis.shape, len(times))


def averaging_algorithm(D, anchors, basis, times, t_window=1.0, n_times=None, verbose=False):
    """ Iteratively compute estimates over fixed time window.

    :param D: measurement matrix with squared distances (N x M)
    :param anchors: anchor coordinates (dim x M)
    :param basis: basis vectors (K x N)
    :param times: list of measurement times (length N)

    :param t_window: width of fixed time window. 

    """
    if type(times) == list:
        times = np.array(times)

    verify_dimensions(D, anchors, basis, times)

    D_k = np.empty((0, D.shape[1]))
    basis_k = np.empty((basis.shape[0], 0))
    C_list = []
    t_list = []

    # make sure we always have overlap
    if n_times is None:
        t_start = np.arange(times[0], times[-1], step=t_window / 2)
    else:
        t_start = np.linspace(times[0], times[-1], n_times)

    for t_s in t_start:
        tk = times[(times >= t_s) & (times < t_s + t_window)]
        for t_n in tk:
            idx = np.where(times == t_n)[0]

            d_mn_row = D[idx, :]
            f_n = basis[:, idx]

            D_k = np.r_[D_k, d_mn_row]
            basis_k = np.c_[basis_k, f_n]

        try:
            C_k = trajectory_recovery(D_k, anchors, basis_k)
            # We need this somehow for numercial reasons.
            # Otherwise sometimes the tests fail because -0. != 0.
            C_k[np.abs(C_k) <= 1e-10] = 0.0
            C_list.append(C_k)
        except AssertionError:
            if verbose:
                print('skipping {:.2f} because only {} measurements.'.format(t_s, len(np.array(tk))))
        except np.linalg.LinAlgError:
            if verbose:
                print('skipping {:.2f} because failed.'.format(t_s))
        except ValueError:
            raise

        D_k = np.empty((0, D.shape[1]))
        basis_k = np.empty((basis.shape[0], 0))
        t_list.append(tk)
    return C_list, t_list


def build_up_algorithm(D, anchors, basis, times, eps=1, verbose=False):
    """ Build-up algorithm for trajectory estimation. 
    
    Build up different trajectories as long as measurements "fit". When they 
    stop fitting (see eps parameter), start a new trajectory.

    :param D: measurement matrix with squared distances (N x M)
    :param anchors: anchor coordinates (dim x M)
    :param basis: basis vectors (K x N)
    :param times: list of measurement times (length N)

    :param eps: error threshold for starting new trajectory.

    """

    verify_dimensions(D, anchors, basis, times)

    C_k = None
    tk = []

    D_k = np.empty((0, D.shape[1]))
    basis_k = np.empty((basis.shape[0], 0))

    C_list = []
    t_list = []

    def g(C_k):
        r_n = C_k.dot(f_n).reshape((ams.shape[0], 1))
        dist_estimates = np.linalg.norm(ams - r_n, axis=0)
        distances = np.sqrt(d_mn_row.flatten()[d_mn_row.flatten() > 0])
        return np.sum(np.abs(dist_estimates - distances)) / len(distances)  # MAE

    for d_mn_row, t_n, f_n in zip(D, times, basis.T):
        d_mn_row = d_mn_row.reshape((1, -1))
        ams = anchors[:, d_mn_row[0] > 0]
        if C_k is None or g(C_k) <= eps:
            D_k = np.r_[D_k, d_mn_row]
            basis_k = np.c_[basis_k, f_n]
            tk.append(t_n)

            try:
                C_test = trajectory_recovery(D_k, anchors, basis_k)
                # TODO find a sensible general threshold here.
                assert g(C_test) < 2 * eps
                C_k = C_test
                # We need this somehow for numercial reasons.
                # Otherwise sometimes the tests fail because -0. != 0.
                C_k[np.abs(C_k) <= 1e-10] = 0.0
            except AssertionError as e:
                if verbose:
                    print('skipping {:.2f} because only {} measurements.'.format(t_n, len(np.array(tk))))
            except np.linalg.LinAlgError:
                if verbose:
                    print('skipping {:.2f} because failed'.format(t_n))

        elif (C_k is not None) and g(C_k) > eps:
            if verbose:
                print('changing to new trajectory, because error is {:.4f}'.format(g(C_k)))

            basis_k = f_n
            D_k = d_mn_row

            C_list.append(C_k)
            t_list.append(tk)

            tk = [t_n]
            C_k = None

    if C_k is not None:
        C_list.append(C_k)
        t_list.append(tk)
    return C_list, t_list


def get_smooth_points(C_list, t_list, traj):
    """ Average the obtained trajectories. """
    result_df = pd.DataFrame(columns=['px', 'py', 't'])
    for Chat, t in zip(C_list, t_list):
        traj.set_coeffs(coeffs=Chat)
        positions = traj.get_sampling_points(times=t)
        this_df = pd.DataFrame({'px': positions[0, :], 'py': positions[1, :], 't': t})
        result_df = pd.concat((this_df, result_df))
    result_df.sort_values('t', inplace=True)
    result_df.reindex()

    import datetime
    mean_window = 10
    datetimes = [datetime.datetime.fromtimestamp(t) for t in result_df.t]
    result_df.index = [pd.Timestamp(datetime) for datetime in datetimes]
    result_df.loc[:, 'px_median'] = result_df['px'].rolling('{}s'.format(mean_window), min_periods=1,
                                                            center=False).median()
    result_df.loc[:, 'py_median'] = result_df['py'].rolling('{}s'.format(mean_window), min_periods=1,
                                                            center=False).median()
    return result_df


def plot_individual(C_list, t_list, traj):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 7)

    for Chat, t in zip(C_list, t_list):
        traj.set_coeffs(coeffs=Chat)
        if len(t) > 0:
            traj.plot(ax=ax, times=t)
            #traj.plot(ax=ax, times=t, label='{:.1f}'.format(t[0]))
    return ax


def plot_smooth(result_df):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 7)
    #plt.scatter(result_df.px, result_df.py, s=1)
    plt.scatter(result_df.px_median, result_df.py_median, s=2, color='red')
    plt.plot(result_df.px_median, result_df.py_median, color='red')
    return ax
