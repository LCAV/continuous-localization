import matplotlib.pyplot as plt
import numpy as np
import os


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
    ''' Plot the cdf of values. 

    :param values: values to plot. 
    :param ax: axis to plot.
    :param kwargs: any other kwargs that are passed to plot function. 
    '''
    probabilities = np.linspace(0, 1, len(values))
    ax.plot(np.sort(values), probabilities, **kwargs)
    ax.yaxis.set_major_locator(plt.FixedLocator(np.linspace(0, 1, 5)))
    ax.set_ylabel('cdf')
    ax.grid(True)


def plot_distances(data_df):
    import itertools
    colors = itertools.cycle(plt.get_cmap('tab10').colors)
    rtt_ids = anchors_df[anchors_df.system_id == rtt_system_id].anchor_id.unique()
    fig, axs = plt.subplots(1, len(rtt_ids), sharey=True)
    fig.set_size_inches(15, 5)
    for ax, anchor_id in zip(axs, rtt_ids):
        color = next(colors)
        data = data_df[data_df.anchor_id == anchor_id]
        anchor_name = anchors_df.loc[anchors_df.anchor_id == anchor_id, 'anchor_name']
        ax.plot(data.seconds, data.distance_gt, linestyle=':', label=anchor_name, color=color)
        ax.plot(data.seconds, data.distance, linestyle='-', color=color)
        ax.set_ylim(0, 15)
