#!/usr/bin/env python

import json
import numpy as np
import scipy as sp

with open("../dykes_data.json") as f:
    moe2017 = json.load(f)

log10_period = moe2017["log10M1"][0]["0.0"]["logP"].keys()
log10_period = np.array([float(key) for key in log10_period])
log10_period_width = 0.5
edges_log10_period = log10_period - log10_period_width/2.
edges_log10_period = np.append(
    edges_log10_period, edges_log10_period[-1] + log10_period_width
)

log10_primary_mass = moe2017["log10M1"][0].keys()
log10_primary_mass = np.array([float(key) for key in log10_primary_mass])
log10_primary_mass_width = 0.1
edges_log10_primary_mass = log10_primary_mass - log10_primary_mass_width/2.
edges_log10_primary_mass = np.append(
    edges_log10_primary_mass,
    edges_log10_primary_mass[-1] + log10_primary_mass_width
)

counts = np.array([
    [moe2017["log10M1"][0][i]["logP"][j]["periodfrac"]
     for j in moe2017["log10M1"][0]["0.0"]["logP"].keys()]
     for i in moe2017["log10M1"][0].keys()
])
cumsum = np.cumsum(counts*log10_period_width, axis=1)
counts /= cumsum[:,-1][:,None]
cumsum /= cumsum[:,-1][:,None]

data = {
    "edges_log10_period": edges_log10_period.tolist(),
    "edges_log10_primary_mass": edges_log10_primary_mass.tolist(),
    "counts": counts.tolist(),
    "cumsum": cumsum.tolist(),
}

with open("data.json", "w") as f:
    json.dump(data, f, indent=2)

# ########################################################################
# # Plot PDFs and CDFs
# ########################################################################

# import plot
# import matplotlib as mpl
# import matplotlib.pyplot as plt

# mpl.style.use("sm")

# fig, ax = plot.plot()
# ax.stairs(counts[0], edges_log10_period, color="red")
# ax.stairs(counts[-1], edges_log10_period, color="magenta")
# plt.savefig("./Figures/log10_period_hist.pdf")
# plt.show()
