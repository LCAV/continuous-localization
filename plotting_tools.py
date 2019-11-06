from itertools import cycle
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import os
import pandas as pd
import seaborn as sns

import simulation  # to avoid circular imports
from global_variables import DIM, RTT_SYSTEM_ID


def make_dirs_safe(path):
    """ Make directory of input path, if it does not exist yet. """
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def savefig(fig, name):
    make_dirs_safe(name)
    fig.savefig(name, bbox_inches='tight', pad_inches=0)


def add_scalebar(ax, size=5, size_vertical=1, loc='lower left'):
    """ Add a scale bar to the plot. 

    :param ax: axis to use.
    :param size: size of scale bar.
    :param size_vertical: height (thckness) of the bar
    :param loc: location (same syntax as for matplotlib legend)
    """
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.font_manager as fm
    fontprops = fm.FontProperties(size=8)
    scalebar = AnchoredSizeBar(ax.transData,
                               size,
                               '{} m'.format(size),
                               loc,
                               pad=0.1,
                               color='black',
                               frameon=False,
                               size_vertical=size_vertical,
                               fontproperties=fontprops)
    ax.add_artist(scalebar)


def remove_ticks(ax):
    """ Remove all ticks and margins from plot. """
    for ax_name in ['x', 'y']:
        ax.tick_params(axis=ax_name,
                       which='both',
                       bottom=False,
                       top=False,
                       left=False,
                       labelbottom=False,
                       labelleft=False)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0.1)
    plt.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())


def plot_cdf_raw(values, ax, **kwargs):
    """Plot the cdf of values.

    :param values: values to plot. 
    :param ax: axis to plot.
    :param kwargs: any other kwargs that are passed to plot function. 
    """
    probabilities = np.linspace(0, 1, len(values))
    ax.plot(np.sort(values), probabilities, **kwargs)
    ax.yaxis.set_major_locator(plt.FixedLocator(np.linspace(0, 1, 5)))
    ax.set_ylabel('cdf')
    ax.grid(True)


def plot_distances(data_df, anchors_df):
    import itertools
    colors = itertools.cycle(plt.get_cmap('tab10').colors)
    rtt_ids = anchors_df[anchors_df.system_id == RTT_SYSTEM_ID].anchor_id.unique()
    fig, axs = plt.subplots(1, len(rtt_ids), sharey=True)
    fig.set_size_inches(15, 5)
    for ax, anchor_id in zip(axs, rtt_ids):
        color = next(colors)
        data = data_df[data_df.anchor_id == anchor_id]
        anchor_name = anchors_df.loc[anchors_df.anchor_id == anchor_id, 'anchor_name']
        ax.plot(data.seconds, data.distance_gt, linestyle=':', label=anchor_name, color=color)
        ax.plot(data.seconds, data.distance, linestyle='-', color=color)
        ax.set_ylim(0, 15)


def plot_noise(key,
               save_figures,
               error_types=None,
               min_noise=None,
               max_noise=None,
               background_alpha=0.1,
               start=1,
               anchors=False,
               lines=None,
               ax=None):
    if error_types is None:
        error_types = ['absolute-errors', 'relative-errors', 'errors']

    if lines is None:
        lines = ["-", "--", "-.", ":"]

    linecycler = cycle(lines)

    resultfolder = 'results/{}/'.format(key)
    results = simulation.read_results(resultfolder + 'result_')
    parameters = simulation.read_params(resultfolder + 'parameters.json')

    min_measurements = (DIM + 2) * parameters["complexities"][0] - 1

    if anchors:
        second_dim = parameters['anchors']
    else:
        second_dim = parameters['noise_sigmas']

    max_measurements = np.min(parameters["positions"]) * np.min(parameters["complexities"])
    if 'sampling_strategy' in parameters:
        if parameters['sampling_strategy'] == 'single_time':
            max_measurements = np.min(parameters["positions"])

    for error_type in error_types:
        error = np.mean(results[error_type], axis=-1)
        print(error.shape)
        error = error.squeeze()
        print(error.shape)
        measurements = np.arange(min_measurements, max_measurements + 1)[::-1]
        if len(second_dim) == 1:
            error = error[:, None]

        if ax is None:
            _, ax1 = plt.subplots()
        else:
            ax1 = ax

        if len(second_dim) == 1:
            error = error.T
        for idx, _ in enumerate(second_dim[min_noise:max_noise]):
            plot = ax1.loglog(measurements, error.T[:len(measurements), idx], alpha=background_alpha)
            z = np.polyfit(np.log(measurements[:-start]), np.log(error[idx, :len(measurements[:-start])]), 1)
            pol = np.poly1d(z)
            print(("anchors {}" if anchors else "noise: {}").format(second_dim[idx]))
            print("fitted slope: {:.2f}".format(z[0]))
            ax1.loglog(measurements,
                       np.exp(pol(np.log(measurements))),
                       c=plot[0].get_color(),
                       label=("{} anchors" if anchors else r"noise: {}").format(second_dim[idx]),
                       linestyle=next(linecycler))

        plt.xlabel("number of measurements")
        if error_type == "errors":
            plt.ylabel("errors")
        else:
            plt.ylabel(" ".join(error_type.split("-") + ["on distances"]))
        ax1.legend(loc='upper right', mode='expand', ncol=len(second_dim[min_noise:max_noise]))
        ax1.set_xticks(measurements[::int(len(measurements) / 5)])
        ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax1.get_xaxis().set_minor_formatter(ticker.NullFormatter())
        plt.grid()
        plt.tight_layout()
        # plt.title(key)
        if save_figures:
            plt.savefig(resultfolder + "oversapling_" + error_type + ".pdf", bbox_inches="tight")
    return plt.gca()


def read_plot_df(name, folder='experiments/robot_test/'):
    """
    read dataset of given name.

    :param name: name of dataset, for example circle2_double.csv
    :return:
        - data_df: calibrated dataset, with columns [distance, distance_mean_0, ...]
        - plot_df: same dataset with one column distance, and one column distance_type.
    """
    datafile = folder + name
    datafile_name = datafile.split('.')[0]
    calibrate_name = datafile_name + '_calibrated.pkl'

    data_df = pd.read_pickle(calibrate_name)

    data_df = data_df.drop(columns=['theta_x', 'theta_y', 'theta_z'], errors='ignore')
    print('read', calibrate_name)

    id_vars = [c for c in data_df.columns if c[:8] != 'distance']
    plot_df = data_df.melt(id_vars=id_vars, var_name="distance_type", value_name="distance")
    if 'RTT' in plot_df.system_id.unique():
        print('Warning: using old RTT system id.')
        plot_df = plot_df[plot_df.system_id == 'RTT']
    elif 'Range' in plot_df.system_id.unique():
        plot_df = plot_df[plot_df.system_id == 'Range']
    else:
        raise NameError('no valid system id in {}'.format(plot_df.system_id.unique()))

    plot_df.sort_values(['timestamp', 'anchor_name'], inplace=True)
    plot_df.reset_index(inplace=True, drop=True)
    plot_df.loc[:, 'distance'] = plot_df.distance.astype(np.float32)
    plot_df.loc[:, 'timestamp'] = plot_df.timestamp.astype(np.float32)

    return data_df, plot_df


def plot_cdfs(plot_df, filename=''):
    colors = sns.color_palette('deep')

    fig, axarr = plt.subplots(2, 4)
    fig.set_size_inches(15, 10)
    axarr = axarr.reshape((-1, ))
    for i, (anchor_name, df) in enumerate(plot_df.sort_values("anchor_name").groupby("anchor_name")):

        axarr[i].set_title(anchor_name)
        df = df.sort_values("timestamp")
        gt_df = df[df.distance_type == "distance_gt"]

        color_cycle = cycle(colors)
        for distance_type in sorted(df.distance_type.unique()):
            if distance_type == "distance_tango" or distance_type == "distance_gt":
                continue
            meas_df = df[df.distance_type == distance_type]

            np.testing.assert_allclose(meas_df.timestamp.values, gt_df.timestamp.values)

            errors = np.abs(meas_df.distance.values - gt_df.distance.values)
            plot_cdf_raw(errors, ax=axarr[i], color=next(color_cycle), label=distance_type)
            axarr[i].set_ylabel('')
    axarr[i].legend(loc='lower left', bbox_to_anchor=[1.0, 0])
    axarr[0].set_ylabel('cdf [-]')
    axarr[4].set_ylabel('cdf [-]')
    for j in range(i + 1, len(axarr)):
        axarr[j].axis('off')
    [axarr[i].set_xlabel('absolute distance error [m]') for i in range(4, len(axarr))]

    if filename != '':
        savefig(fig, filename)


def plot_times(plot_df, filename=''):
    colors = sns.color_palette('deep')

    fig, axarr = plt.subplots(2, 4)
    fig.set_size_inches(15, 10)
    axarr = axarr.reshape((-1, ))
    for i, (anchor_name, df) in enumerate(plot_df.sort_values("anchor_name").groupby("anchor_name")):
        axarr[i].set_title(anchor_name)
        df = df.sort_values("timestamp")

        color_cycle = cycle(colors)
        for distance_type in sorted(df.distance_type.unique()):
            meas_df = df[df.distance_type == distance_type]
            axarr[i].plot(meas_df.timestamp.values,
                          meas_df.distance.values,
                          color=next(color_cycle),
                          label=distance_type)
    axarr[i].legend(loc='lower left', bbox_to_anchor=[1.0, 0])
    for j in range(i + 1, len(axarr)):
        axarr[j].axis('off')

    if filename != '':
        savefig(fig, filename)


def plot_rssis(plot_df, filename=''):
    fig, axarr = plt.subplots(2, 4, sharey=True, sharex=True)
    fig.set_size_inches(15, 10)
    axarr = axarr.reshape((-1, ))
    for i, (anchor_name, df) in enumerate(plot_df.sort_values("anchor_name").groupby("anchor_name")):
        axarr[i].set_title(anchor_name)
        df = df.sort_values("timestamp")
        gt_df = df[df.distance_type == "distance_gt"]

        meas_df = df[df.distance_type == "distance_median_all"]

        assert np.allclose(meas_df.timestamp.values, gt_df.timestamp.values)

        errors = np.abs(meas_df.distance.values - gt_df.distance.values)
        rssis = meas_df.rssi.values

        axarr[i].scatter(rssis, errors, alpha=0.2)
    for j in range(i + 1, len(axarr)):
        axarr[j].axis('off')

    if filename != '':
        savefig(fig, filename)


def plot_tango_components(data_df, filename=''):
    fig = plt.figure()
    data = data_df[data_df.system_id == "GT"]
    sns.scatterplot(data=data, x='timestamp', y='px', linewidth=0.0, label='x')
    sns.scatterplot(data=data, x='timestamp', y='py', linewidth=0.0, label='y')
    sns.scatterplot(data=data, x='timestamp', y='pz', linewidth=0.0, label='z')
    plt.legend()

    if filename != '':
        savefig(fig, filename)


def plot_tango_2d(data_df, anchors_df, filename=''):
    tango_df = data_df.loc[data_df.system_id == "GT"]

    fig = plt.figure()
    sns.scatterplot(data=tango_df, x='px', y='py', hue='timestamp', linewidth=0.0)
    sns.scatterplot(data=anchors_df, x='px', y='py', hue='anchor_name', linewidth=0.0, style='system_id', legend=False)
    plt.arrow(0, 0, 1, 0, color='red', head_width=0.2)
    plt.arrow(0, 0, 0, 1, color='green', head_width=0.2)
    delta = 0.1
    for i, a in anchors_df.iterrows():
        plt.annotate(a.anchor_name, (a.px + delta, a.py + delta))
    plt.legend()
    plt.axis('equal')
    if filename != '':
        savefig(fig, filename)


#TODO(FD) below too functions have too much computation. Try to split
# plotting and processing more cleanly.
def plot_subsample(traj, D, times, anchors, full_df, n_measurements_list):
    import time

    import hypothesis as h
    from other_algorithms import pointwise_srls
    from solvers import alternativePseudoInverse

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

            Chat = alternativePseudoInverse(D_small, anchors[:2, :], basis_small, weighted=True)

            coeffs = np.dstack([coeffs, Chat])
            traj.set_coeffs(coeffs=Chat)

            traj.plot_pretty(times=times, color='C{}'.format(colors[seed]), ax=ax, alpha=alpha)

        Chat_avg = np.mean(coeffs, axis=2)
        traj.set_coeffs(coeffs=Chat_avg)
        traj.plot_pretty(times=times, color='C0', label=label, ax=ax)

        points = pointwise_srls(D, anchors, basis, traj, indices)
        label = 'SRLS'
        for x in points:
            ax.scatter(*x, color='C1', label=label, s=4.0)
            label = None

        ax.plot(full_df.px, full_df.py, ls=':', linewidth=1., color='black', label='GPS')
        remove_ticks(ax)
        ax.set_title('N = {}'.format(n_measurements), y=-0.22)
    add_scalebar(axs[0], 20, loc='lower left')
    return fig, axs


def plot_complexities(traj, D, times, anchors, full_df, list_complexities, srls=True):
    from other_algorithms import pointwise_srls
    from solvers import alternativePseudoInverse

    fig, axs = plt.subplots(1, len(list_complexities), sharex=True, sharey=True)
    for ax, n_complexity in zip(axs, list_complexities):
        traj.set_n_complexity(n_complexity)
        basis = traj.get_basis(times=times)

        Chat = alternativePseudoInverse(D, anchors[:2, :], basis, weighted=True)

        traj.set_coeffs(coeffs=Chat)

        traj.plot_pretty(times=times, color="C0", label='ours', ax=ax)
        ax.plot(full_df.px, full_df.py, color='black', ls=':', linewidth=1., label='GPS')
        ax.set_title('K = {}'.format(traj.n_complexity))

        remove_ticks(ax)
        if srls:
            indices = range(D.shape[0])[::3]
            points = pointwise_srls(D, anchors, basis, traj, indices)
            label = 'SRLS'
            for x in points:
                ax.scatter(*x, color='C1', label=label, s=4.0)
                label = None

    add_scalebar(axs[0], 20, loc='lower left')
    return fig, axs


def plot_probabilities(
        ranks,
        params,
        directory="results/ranks/",
        save=False,
):

    key = "_d{}_c{}_{}_full{}".format(params["n_dimensions"], params["n_constraints"], params["second_key"],
                                      params["full_matrix"])

    max_rank = params["max_rank"]
    n_repetitions = ranks.shape[2]
    x = np.array(params["second_list"])
    if "fixed_n_measurements" not in params:
        x = x / max_rank

    f, ax = plt.subplots()
    for a_idx, n_anchors in enumerate(params["n_anchors_list"]):
        plt.plot(x,
                 np.mean(ranks[:, a_idx, :], axis=1) / max_rank,
                 label="mean rank, {} anchors".format(n_anchors),
                 color="C{}".format(a_idx),
                 linestyle='dashed')
        plt.step(x,
                 np.sum(ranks[:, a_idx, :] >= max_rank, axis=1) / n_repetitions,
                 label="probability, {} anchors".format(n_anchors),
                 color="C{}".format(a_idx),
                 where='post')
    if "fixed_n_measurements" in params:
        plt.xlabel("number of positions")
    else:
        plt.xlabel("number of measurements")
        formatter_text = '%g (D+1)K + (K-1)' if params["full_matrix"] else '%g (D+1)K'
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(formatter_text))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(base=1))
    plt.grid()
    plt.legend()
    params["directory"] = directory
    if save:
        plt.ylim(bottom=0)
        matrix_type = "full" if params["full_matrix"] else "left"
        fname = directory + matrix_type + "_matrix_anchors" + key + ".pdf"
        make_dirs_safe(fname)
        plt.savefig(fname)
