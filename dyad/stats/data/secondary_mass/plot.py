#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import quad

mpl.style.use("sm")
cmap = "viridis"

m_min = 0.08
m_max = 60.
primary_mass_boundary = (m_min, 1.2, 3.5, 6., m_max)

mass_ratio_sample = np.loadtxt("mass_ratio_sample.dat")
primary_mass_sample = np.loadtxt("primary_mass_sample.dat")
pdf_sample = np.loadtxt("frequency_sample.dat").T
cdf_sample = np.loadtxt("cumulative_frequency_sample.dat").T

########################################################################
# Plot PDF
########################################################################
# fig, ax = plt.subplots()
# ax.pcolormesh(
#     primary_mass_sample, mass_ratio_sample,
#     pdf_sample, cmap=cmap, rasterized=True
# )
# ax.contour(
#     primary_mass_sample, mass_ratio_sample,
#     np.log10(pdf_sample), colors="k", levels=10
# )
# ax.hlines(0.1, m_min, m_max, color="k", ls="dashed")
# ax.hlines(0.3, m_min, m_max, color="k", ls="dashed")
# ax.hlines(0.95, m_min, m_max, color="k", ls="dashed")
# ax.vlines(primary_mass_boundary, 0., 1., ls="dashed")
# ax.set_xlim(m_min, m_max)
# ax.set_ylim(0., 1.)
# ax.set_xscale("log")
# ax.set_xlabel(r"$m_{1}$")
# ax.set_ylabel(r"$q$")
# fig.savefig("./Figures/f_QgivenM1_moe2017_pairing_2d.pdf", dpi=300)
# fig.savefig("./Figures/f_QgivenM1_moe2017_pairing_2d.jpg", dpi=300)
# fig.show()

# fig, ax = plt.subplots()
# ax.pcolormesh(
#     primary_mass_sample, mass_ratio_sample,
#     np.log10(cdf_sample),
#     cmap=cmap, rasterized=True
# )
# ax.contour(
#     primary_mass_sample, mass_ratio_sample,
#     np.log10(cdf_sample), colors="k",
#     levels=10
# )
# ax.hlines(0.1, m_min, m_max, color="k", ls="dashed")
# ax.hlines(0.3, m_min, m_max, color="k", ls="dashed")
# ax.hlines(0.95, m_min, m_max, color="k", ls="dashed")
# ax.vlines(primary_mass_boundary, 0., 1., ls="dashed")
# ax.set_xlim(m_min, m_max)
# ax.set_ylim(0., 1.)
# ax.set_xscale("log")
# ax.set_xlabel(r"$m_{1}$")
# ax.set_ylabel(r"$q$")
# fig.savefig("./Figures/F_QgivenM1_moe2017_pairing_2d.pdf", dpi=300)
# fig.savefig("./Figures/F_QgivenM1_moe2017_pairing_2d.jpg", dpi=300)
# plt.show()

########################################################################
# Plot CDF
########################################################################
# fig, ax = plt.subplots()
# for m, f_m in zip(primary_mass_sample[::50], pdf_sample.T[::50]):
#     ax.plot(mass_ratio_sample, f_m,
#             label=r"$m_{{1}} = {:1.3f}~\mathrm{{M}}_{{\odot}}$".format(m))
# ax.set_xlim(0., 1.)
# ax.set_ylim(0., 5.)
# ax.set_xlabel(r"$q$")
# ax.set_ylabel(r"$f_{Q}(\cdot|m_{1})$")
# ax.legend(frameon=False)
# fig.savefig("./Figures/f_QgivenM1_moe2017_pairing.pdf", dpi=300)
# fig.savefig("./Figures/F_QgivenM1_moe2017_pairing.jpg", dpi=300)
# plt.show()

# fig, ax = plt.subplots()
# for m, F_m in zip(primary_mass_sample[::50], cdf_sample.T[::50]):
#     ax.plot(mass_ratio_sample, F_m, ls="solid",
#             label=r"$m_{{1}} = {:1.3f}~\mathrm{{M}}_{{\odot}}$".format(m))
# ax.set_xlim(0., 1.)
# ax.set_ylim(0., 1.)
# ax.set_xlabel(r"$q$")
# ax.set_ylabel(r"$f_{Q}(\cdot|m_{1})$")
# ax.legend(frameon=False)
# fig.savefig("./Figures/F_QgivenM1_moe2017_pairing.pdf", dpi=300)
# fig.savefig("./Figures/F_QgivenM1_moe2017_pairing.jpg", dpi=300)
# plt.show()

########################################################################
# Form the pairing function
########################################################################
def f_secondary(m_2, m_1=1.):
    q = m_2/m_1
    res = pdf_interp((q, m_1))/m_1

    return res
    
pdf_interp = RegularGridInterpolator(
    (mass_ratio_sample, primary_mass_sample),
    pdf_sample,
    bounds_error=False,
    fill_value=0.
)
cdf_interp = RegularGridInterpolator(
    (mass_ratio_sample, primary_mass_sample),
    cdf_sample,
    bounds_error=False,
    fill_value=0.
)

########################################################################
# Plot the pairing function
########################################################################
n = 1_000
m_1 = np.hstack([
    np.linspace(m_min, 1.2, n),
    np.linspace(1.2, 3.5, n)[1:],
    np.linspace(3.5, 6., n)[1:],
    np.linspace(6., m_max, n)[1:],
])
m1m1, m2m2 = np.meshgrid(m_1, m_1)
f_M2 = f_secondary(m2m2, m1m1)
fig, ax = plt.subplots()
for i, f_i in zip(m_1[::1_000], f_M2[::1_000]):
    ax.plot(
        m_1,
        f_i/np.max(f_i),
        ls="solid",
        label=r"$m_{{1}} = {:1.3f}~\mathrm{{M}}_{{\odot}}$".format(i)
    )
    I = quad(f_secondary, m_min, m_max)
    print(I)
# ax.set_ylim(0., 5.)
ax.set_xscale("log")
ax.set_xlabel(r"$q$")
ax.set_ylabel(r"$f_{Q}(\cdot|m_{1})$")
ax.legend(frameon=False)
fig.savefig("./Figures/F_QgivenM1_moe2017_pairing.pdf", dpi=300)
fig.savefig("./Figures/F_QgivenM1_moe2017_pairing.jpg", dpi=300)
plt.show()

