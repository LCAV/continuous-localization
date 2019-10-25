# coding: utf-8
"""
This script contains experiments are probably not going to be used in the paper,
because they use many measurements per time.

They can be used as examples or removed later.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

import hypothesis as h

plt.rcParams['figure.figsize'] = 10, 5

# Rank vs number of positions
# Here the number of dimentions $D$, the number of constrains $K$ and the total
# number of measrurements are fixed. In particular, the number of measurements is $(D+1)K$, and we increase total
# number of sampling positions along the trajectory. We can see drastically different behaviour for exactly $D+1$
# anchors than for more than $D+1$ anchors.
print("Starting simple simulations")

experiment_params = {
    "n_dimensions": 2,
    "n_constraints": 5,
    "fixed_n_measurements": 0,
    "max_positions": 100,
    "n_repetitions": 1000,
    "full_matrix": True,
    "n_anchors_list": [1, 2, 4, 8, 100],
}

start = time.time()
ranks, params = h.matrix_rank_experiment(experiment_params)

h.plot_results(ranks, params, save=True)
plt.show()
end = time.time()
print("First simulations, elapsed time: {:.2f}s".format(end - start))

# Rank vs number of positions
# Same a above, but with twice oversampling

experiment_params = {
    "n_dimensions": 2,
    "n_constraints": 5,
    "fixed_n_measurements": 30,
    "max_positions": 100,
    "n_repetitions": 1000,
    "full_matrix": True,
    "n_anchors_list": [1, 2, 4, 8, 100],
}

start = time.time()
ranks, params = h.matrix_rank_experiment(experiment_params)
end = time.time()
print("Oversampled simulations, elapsed time: {:.2f}s".format(end - start))

h.plot_results(ranks, params, save=True)
plt.show()

# Theory vs experiments
# This part compares rank of the left hand side of the matrix with theoretical bounds
# Generate experimental data for small number of anchors
# (the exact calculations are expensive for large number of anchors)

print("Start comparision experiments")
n_constrains = 5
experiment_params = {
    "n_dimensions": 2,
    "n_constraints": n_constrains,
    "fixed_n_measurements": 0,
    "max_positions": 100,
    "n_repetitions": 1000,
    "full_matrix": False,
    "n_anchors_list": [1, 2, 3, 10],
}

start = time.time()
ranks, params = h.matrix_rank_experiment(experiment_params)

h.plot_results(ranks, params)
plt.show()
end = time.time()
print("Comparison experiments done, elapsed time: {:.2f}s".format(end - start))

n_positions_list = params["second_list"]

# Generate the theoretical results for fixed number of anchors
idx = 1
n_anchors = params["n_anchors_list"][idx]
print("Start comparision calculation for {} anchors".format(n_anchors))

probabilities1 = [
    h.probability_upper_bound_any_measurements(params["n_dimensions"],
                                               params["n_constraints"],
                                               n,
                                               position_wise=False,
                                               n_anchors=n_anchors,
                                               n_measurements=params["fixed_n_measurements"]) for n in n_positions_list
]

probabilities2 = [
    h.probability_upper_bound_any_measurements(params["n_dimensions"],
                                               params["n_constraints"],
                                               n,
                                               position_wise=True,
                                               n_anchors=n_anchors,
                                               n_measurements=params["fixed_n_measurements"]) for n in n_positions_list
]

probabilities1 = np.array(probabilities1)
probabilities2 = np.array(probabilities2)

print("Done comparision")

# Plot all the theoretical and estimated bounds
max_rank = params["max_rank"]
mean = np.mean(ranks[:, idx, :] >= max_rank, axis=-1)
std = np.std(ranks[:, idx, :] >= max_rank, axis=-1)
anchor_condition = np.mean(params["anchor_condition"][:, idx, :], axis=-1)
frame_condition = np.mean(params["frame_condition"][:, idx, :], axis=-1)
both_conditions = np.mean((params["anchor_condition"] * params["frame_condition"])[:, idx, :], axis=-1)

f, ax = plt.subplots()
beg = 0

plt.plot(n_positions_list[beg:], probabilities1[beg:], linestyle=":", c="C1", label="anchor upper bound")
plt.plot(n_positions_list[beg:], probabilities2[beg:], linestyle=":", c="C2", label="sample upper bound")
plt.plot(n_positions_list[beg:], probabilities2[beg:] * probabilities1, linestyle=":", c="C3", label="both bounds")
plt.plot(n_positions_list[beg:],
         probabilities2[beg:] * np.min(probabilities1[beg:]),
         linestyle=":",
         c="C0",
         label="a hack")
plt.plot(n_positions_list[beg:], anchor_condition, c="C2", label="sample condition")
plt.plot(n_positions_list[beg:], frame_condition, c="C1", label="anchor condition")
plt.plot(n_positions_list, both_conditions, c="C3", label="both conditions")
plt.plot(n_positions_list[beg:], mean[beg:], c="C0", label="estimated probability")

plt.xlabel("number of positions")
plt.ylim(0)
plt.grid()
plt.legend()
plt.title("Upper bounds for D + {} anchors (with hacks)".format(n_anchors - params["n_dimensions"]))
plt.show()

# Plot the (hacked) full rank probability for both number of anchors and number of times
# Only for the left hand side of the matrix

n_anchors_list = np.arange(3, 31)
n_positions_list = np.arange(5, 31)
n_measurements = params["fixed_n_measurements"]
probabilities = []
probabilities2 = []
for n_anchors in n_anchors_list:
    probabilities.append([
        h.probability_upper_bound_any_measurements(params["n_dimensions"],
                                                   params["n_constraints"],
                                                   n,
                                                   position_wise=False,
                                                   n_anchors=n_anchors,
                                                   n_measurements=n_measurements) for n in n_positions_list
    ])
for n_anchors in n_anchors_list:
    probabilities2.append([
        h.probability_upper_bound_any_measurements(params["n_dimensions"],
                                                   params["n_constraints"],
                                                   n,
                                                   position_wise=True,
                                                   n_anchors=n_anchors,
                                                   n_measurements=n_measurements) for n in n_positions_list
    ])

fig, ax = plt.subplots(figsize=(6, 6))
probabilities = np.array(probabilities)
probabilities2 = np.array(probabilities2)
im = ax.imshow((probabilities + probabilities2 - 1))
ax.set_xticks(n_positions_list[::2] - np.min(n_positions_list))
ax.set_yticks(n_anchors_list[::2] - np.min(n_anchors_list))
ax.set_xticklabels(n_positions_list[::2])
ax.set_yticklabels(n_anchors_list[::2])
plt.xlabel("# positions")
plt.ylabel("# anchros")
plt.colorbar(im)
plt.title("success probability (upper bound)")
plt.show()

# Inspect matrices that satisfy both conditions but are not full rank.
answer = input("Do you want to inspect matrices of the wrong rank?")
if answer in ['y', 'yes']:
    df = pd.DataFrame(params["wrong_matrices"])
    for r in range(110, 120):
        row = dict(df.loc[r, :])
        print("anchors:", params["n_anchors_list"][row["a_idx"]])
        _, sv, _ = np.linalg.svd(row["constraints"])
        plt.semilogy(np.arange(1, len(sv) + 1), sv)
        plt.plot()
        plt.matshow(row["constraints"])
        plt.colorbar()
        plt.show()
        print(row["measurements"])
