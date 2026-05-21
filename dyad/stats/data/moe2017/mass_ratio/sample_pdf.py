#!/usr/bin/env python

import json
import numpy as np
import scipy as sp

# Load JSON file containing data of Moe and Di Stefano (2017)
with open("../dykes_data.json") as f:
    moe2017 = json.load(f)

# Find edges of mass-ratio bins
mass_ratio = moe2017["log10M1"][0]["0.0"]["logP"]["0.25"]["q"].keys()
mass_ratio = np.array([float(key) for key in mass_ratio])
mass_ratio_width = 0.1
edges_mass_ratio = mass_ratio - mass_ratio_width/2.
edges_mass_ratio = np.append(
    edges_mass_ratio, edges_mass_ratio[-1] + mass_ratio_width
)

# Find edges of log-period bins
log10_period = moe2017["log10M1"][0]["0.0"]["logP"].keys()
log10_period = np.array([float(key) for key in log10_period])
log10_period_width = np.diff(log10_period)[0]
edges_log10_period = log10_period - log10_period_width/2.
edges_log10_period = np.append(
    edges_log10_period, edges_log10_period[-1] + log10_period_width
)

# Find edges of log-primary mass bins
log10_primary_mass = moe2017["log10M1"][0].keys()
log10_primary_mass = np.array([float(key) for key in log10_primary_mass])
log10_primary_mass_width = np.diff(log10_primary_mass)[0]
edges_log10_primary_mass = log10_primary_mass - log10_primary_mass_width/2.
edges_log10_primary_mass = np.append(
    edges_log10_primary_mass,
    edges_log10_primary_mass[-1] + log10_primary_mass_width
)

# Find bin counts
key_log10_primary_mass = moe2017["log10M1"][0].keys()
key_log10_period = moe2017["log10M1"][0]["0.0"]["logP"].keys()
key_mass_ratio = moe2017["log10M1"][0]["0.0"]["logP"]["0.25"]["q"].keys()
counts = np.zeros(
    [len(log10_primary_mass), len(log10_period), len(mass_ratio)]
)
for i, key_i in enumerate(key_log10_primary_mass):
    for j, key_j in enumerate(key_log10_period):
        for k, key_k in enumerate(key_mass_ratio):
            counts[i][j][k] = (
                moe2017["log10M1"][0][key_i]["logP"][key_j]["q"][key_k]
            )

# Normalize histogram of mass_ratio
cumsum = np.cumsum(counts*mass_ratio_width, axis=2)
counts /= cumsum[:,:,-1][:,:,None]
cumsum /= cumsum[:,:,-1][:,:,None]

# Save data to JSON file
data = {
    "edges_mass_ratio": edges_mass_ratio.tolist(),
    "edges_log10_period": edges_log10_period.tolist(),
    "edges_log10_primary_mass": edges_log10_primary_mass.tolist(),
    "counts": counts.tolist(),
    "cumsum": cumsum.tolist(),
}
with open("moe2017.json", "w") as f:
    json.dump(data, f, indent=2)

########################################################################
# Plot PDFs and CDFs
########################################################################

import plot
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use("sm")

fig, ax = plot.plot()
ax.stairs(counts[0][2], edges_mass_ratio, color="red")
ax.stairs(counts[-1][2], edges_mass_ratio, color="magenta")
plt.savefig("./Figures/mass_ratio_0_hist.pdf")
plt.show()

fig, ax = plot.plot()
ax.stairs(counts[0][-1], edges_mass_ratio, color="red")
ax.stairs(counts[-1][-1], edges_mass_ratio, color="magenta")
plt.savefig("./Figures/mass_ratio_1_hist.pdf")
plt.show()
