#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.filters import gaussian_filter

primary_mass_extended_sample = np.loadtxt("./primary_mass_extended_sample.dat")
mass_ratio_extended_sample = np.loadtxt("./mass_ratio_extended_sample.dat")
pdf_extended_sample = np.loadtxt("./frequency_extended_sample.dat")
cdf_extended_sample = np.loadtxt("./cumulative_frequency_extended_sample.dat")

primary_mass_boundary = (0.08, 0.8, 1.2, 3.5, 6., 40., 150.)
mass_ratio_boundary = (0., 0.1, 0.3, 0.95, 1.)

mpl.style.use("sm")

fig, ax = plt.subplots()
ax.pcolormesh(
    mass_ratio_extended_sample, primary_mass_extended_sample,
    pdf_extended_sample, rasterized=True, norm="log", vmin=0.1, vmax=3.
)
ax.contour(
    mass_ratio_extended_sample, primary_mass_extended_sample,
    pdf_extended_sample, colors="k", norm="log",
    levels=np.logspace(-3., 3., 25)
)
ax.plot(mass_ratio_extended_sample, 0.08/mass_ratio_extended_sample)
ax.add_patch(
    patches.Rectangle(
        (0.1, 0.8), 0.9, 40 - 0.8, edgecolor="k", facecolor="None", ls="dashed"
    )
)
ax.vlines(mass_ratio_boundary, 0.08, 150., colors="red")
ax.hlines(primary_mass_boundary, 0., 1., colors="red")
ax.set_xlim(0., 1.)
ax.set_ylim(0.08, 150.)
ax.set_yscale("log")
ax.set_xlabel(r"$q$")
ax.set_ylabel(r"$m_{1}/\mathrm{M}_{\odot}$")
plt.savefig("./Figures/f_QgivenM1_moe2017_extended.pdf", dpi=300)
plt.savefig("./Figures/f_QgivenM1_moe2017_extended.jpg", dpi=300)
plt.show()

# fig, ax = plt.subplots()
# ax.pcolormesh(
#     mass_ratio_extended_sample, primary_mass_extended_sample,
#     cdf_extended_sample, rasterized=True, norm="log",
#     vmin=1.e-2, vmax=1.
# )
# ax.contour(
#     mass_ratio_extended_sample, primary_mass_extended_sample,
#     cdf_extended_sample, colors="k", norm="log",
#     levels=np.logspace(-2., 0., 25)[1:]
# )
# ax.plot(mass_ratio_extended_sample, 0.08/mass_ratio_extended_sample)
# # ax.vlines(mass_ratio_boundary, 0.08, 150.)
# # ax.hlines(primary_mass_boundary, 0., 1.)
# ax.set_xlim(0., 1.)
# ax.set_ylim(0.08, 150.)
# ax.set_yscale("log")
# ax.set_xlabel(r"$q$")
# ax.set_ylabel(r"$m_{1}/\mathrm{M}_{\odot}$")
# plt.savefig("./Figures/F_QgivenM1_moe2017_extended.pdf", dpi=300)
# plt.savefig("./Figures/F_QgivenM1_moe2017_extended.jpg", dpi=300)
# plt.show()

# qq, m1m1 = np.meshgrid(
#     mass_ratio_extended_sample, primary_mass_extended_sample
# )

# fig, ax = plt.subplots()
# ax.pcolormesh(
#     mass_ratio_extended_sample, primary_mass_extended_sample,
#     f, rasterized=True, norm="log",
#     vmin=0.1, vmax=5960.
# )
# ax.contour(
#     mass_ratio_extended_sample, primary_mass_extended_sample,
#     np.log10(f), colors="k", levels=np.linspace(0.1, 10., 50)
# )
# ax.plot(mass_ratio_extended_sample, 0.08/mass_ratio_extended_sample)
# # ax.vlines(mass_ratio_boundary, 0.08, 150.)
# # ax.hlines(primary_mass_boundary, 0., 1.)
# ax.set_xlim(0., 1.)
# ax.set_ylim(0.08, 150.)
# ax.set_yscale("log")
# ax.set_xlabel(r"$q$")
# ax.set_ylabel(r"$m_{1}/\mathrm{M}_{\odot}$")
# plt.savefig("./Figures/F_QgivenM1_moe2017.pdf", dpi=300)
# plt.savefig("./Figures/F_QgivenM1_moe2017.jpg", dpi=300)
# plt.show()
