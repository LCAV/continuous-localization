import matplotlib.pyplot as plt


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
