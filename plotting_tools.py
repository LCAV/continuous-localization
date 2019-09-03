from itertools import cycle
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import os
import pandas as pd
import seaborn as sns

from simulation import read_results, read_params
from global_variables import DIM, RTT_SYSTEM_ID


def make_dirs_safe(path):
    """ Make directory of input path, if it does not exist yet. """
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def savefig(fig, name):
    fig.savefig(name, bbox_inches='tight')
    print('saved as', name)


def generate_labels(dims, parameters, indices):
    title = ""
    label = []
    for key, value in dims.items():
        if len(indices[value]) == 1:
            title += ", {}: {}".format(key, parameters[key][indices[value][0]])
        else:
            label.append(key)
    return title, label


def add_plot_decoration(label, parameters):
    plt.colorbar()
    if label[1] != 'measurements':
        x_val = parameters[label[1]]
        plt.yticks(range(len(x_val)), x_val)
        plt.xlabel(label[1])
    else:
        plt.xlabel('number of missing measurements')
    plt.ylabel(label[0])
    y_val = parameters[label[0]]
    plt.yticks(range(len(y_val)), y_val)
    plt.gca().xaxis.tick_bottom()


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


def plot_noise(key, save_figures, error_types=None, min_noise=None, max_noise=None, smoothing=100,
               background_alpha=0.1):
    if error_types is None:
        error_types = ['absolute-errors', 'relative-errors', 'errors']

    resultfolder = 'results/{}/'.format(key)
    results = read_results(resultfolder + 'result_')
    parameters = read_params(resultfolder + 'parameters.json')

    min_measurements = (DIM + 2) * parameters["complexities"][0] - 1
    noise_sigmas = parameters['noise_sigmas']

    x = np.linspace(-smoothing, smoothing, smoothing)
    sinc = np.sinc(x)
    sinc = sinc / np.sum(sinc)

    for error_type in error_types:
        error = results[error_type].squeeze()
        dimensions = error.shape
        measurements = np.arange(min_measurements, dimensions[1])
        new_error = []
        for idx in range(dimensions[0]):
            new_error.append(np.convolve(error[idx, :len(measurements)], sinc, "valid"))
        new_error = np.array(new_error)
        shift = dimensions[1] - len(new_error[0, :]) + 1
        new_measurements = measurements[-shift // 2:shift // 2:-1]
        fig1, ax1 = plt.subplots()
        for idx, _ in enumerate(noise_sigmas[min_noise:max_noise]):
            plot = plt.loglog(
                new_measurements, new_error.T[:len(new_measurements), idx], label="noise: {}".format(noise_sigmas[idx]))
            plt.loglog(
                measurements[::-1], error.T[:len(measurements), idx], alpha=background_alpha, c=plot[0].get_color())
        plt.xlabel("number of measurements")
        if error_type == "errors":
            plt.ylabel("errors on coefficients")
        else:
            plt.ylabel(" ".join(error_type.split("-") + ["on distances"]))
        ax1.legend(loc='center left', bbox_to_anchor=(1., 0.5))
        ax1.set_xticks(measurements[::int(len(measurements) / 5)])
        ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax1.get_xaxis().set_minor_formatter(ticker.NullFormatter())
        plt.grid()
        plt.title(key)
        if save_figures:
            plt.savefig(resultfolder + "oversapling_" + error_type + ".pdf", bbox_inches="tight")
        plt.show()


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
    plot_df = plot_df[plot_df.system_id == 'RTT']
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
        gt_df = df[df.distance_type == "distance_tango"]

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
            axarr[i].plot(
                meas_df.timestamp.values, meas_df.distance.values, color=next(color_cycle), label=distance_type)
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
        gt_df = df[df.distance_type == "distance_tango"]

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
    data = data_df[data_df.system_id == "Tango"]
    sns.scatterplot(data=data, x='timestamp', y='px', linewidth=0.0, label='x')
    sns.scatterplot(data=data, x='timestamp', y='py', linewidth=0.0, label='y')
    sns.scatterplot(data=data, x='timestamp', y='pz', linewidth=0.0, label='z')
    plt.legend()

    if filename != '':
        savefig(fig, filename)


def plot_tango_2d(data_df, anchors_df, filename=''):
    tango_df = data_df.loc[data_df.system_id == "Tango"]

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
