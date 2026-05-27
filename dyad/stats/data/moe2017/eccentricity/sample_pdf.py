#!/usr/bin/env python

import json
import numpy as np
import scipy as sp

from importlib.resources import files, as_file

# # Load JSON file containing data of Moe and Di Stefano (2017)
path = "dyad.stats.data.moe2017"
with as_file(files(path).joinpath("dykes_data.json")) as path:
    with path.open("r") as f:
        moe2017 = json.load(f)
    
# Find edges of eccentricity bins
eccentricity = moe2017["log10M1"][0]["0.0"]["logP"]["0.25"]["e"].keys()
eccentricity = np.array([float(key) for key in eccentricity])
eccentricity_width = 0.1
edges_eccentricity = eccentricity - eccentricity_width/2.
edges_eccentricity[0] = 0.
edges_eccentricity = np.append(
    edges_eccentricity, edges_eccentricity[-1] + eccentricity_width
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
key_eccentricity = moe2017["log10M1"][0]["0.0"]["logP"]["0.25"]["e"].keys()
counts = np.zeros(
    [len(log10_primary_mass), len(log10_period), len(eccentricity)]
)
for i, key_i in enumerate(key_log10_primary_mass):
    for j, key_j in enumerate(key_log10_period):
        for k, key_k in enumerate(key_eccentricity):
            counts[i][j][k] = (
                moe2017["log10M1"][0][key_i]["logP"][key_j]["e"][key_k]
            )

# Normalize histogram of eccentricity
cumsum = np.cumsum(counts*eccentricity_width, axis=2)
counts /= cumsum[:,:,-1][:,:,None]
cumsum /= cumsum[:,:,-1][:,:,None]

# Save data to JSON file
data = {
    "edges_eccentricity": edges_eccentricity.tolist(),
    "edges_log10_period": edges_log10_period.tolist(),
    "edges_log10_primary_mass": edges_log10_primary_mass.tolist(),
    "counts": counts.tolist(),
    "cumsum": cumsum.tolist(),
}
path = "dyad.stats.data.moe2017.eccentricity"
with as_file(files(path).joinpath("data.json")) as path:
    with path.open("w") as f:
        json.dump(data, f, indent=2)

########################################################################
# Plot PDFs and CDFs
########################################################################

import plot
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use("sm")

fig, ax = plot.plot()
ax.stairs(counts[0][2], edges_eccentricity, color="red")
ax.stairs(counts[-1][2], edges_eccentricity, color="magenta")
plt.savefig("./Figures/eccentricity_0_hist.pdf")
plt.show()

fig, ax = plot.plot()
ax.stairs(counts[0][-1], edges_eccentricity, color="red")
ax.stairs(counts[-1][-1], edges_eccentricity, color="magenta")
plt.savefig("./Figures/eccentricity_1_hist.pdf")
plt.show()
