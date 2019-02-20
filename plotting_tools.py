import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_point_with_name(point, name, **kwargs):
    """ Plot points with labels (used for anchors)."""
    ax = plt.gca()
    fig = plt.gcf()
    size = fig.get_size_inches()
    delta = max(*size) / 50.0
    ax.scatter(point[0], point[1], **kwargs)

    ax.annotate(name, (point[0] + delta, point[1] + delta), size=delta * 100, **kwargs)


def add_colorbar(fig, ax, im):
    """ Add colorbar of same size as image. """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad='1%')
    fig.colorbar(im, cax=cax, orientation='vertical')


def add_plot_decoration(ax, xlabel, ylabel, parameters):
    """ Add a good amount of xticks, yticks to matshow, and labels in good positions """
    ax.set_xlabel(xlabel)
    x_val = parameters[xlabel]
    skip = 1
    if len(x_val) > 10:
        skip = int(len(x_val) / 10)
    ax.set_xticks(np.arange(len(x_val))[::skip])
    ax.set_xticklabels(x_val[::skip])
    ax.xaxis.tick_bottom()

    ax.set_ylabel(ylabel)
    y_val = parameters[ylabel]
    ax.set_yticks(range(len(y_val)))
    ax.set_yticklabels(y_val)


def get_n_colors(n):
    cmap = plt.cm.get_cmap("rainbow")
    indices = np.linspace(0, 1, n)
    return cmap(indices)
