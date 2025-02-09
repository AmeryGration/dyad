#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator

mpl.style.use("sm")

m_min = 0.08
m_max = 60.

mass_ratio_sample = np.loadtxt("mass_ratio_sample.dat")
primary_mass_sample = np.loadtxt("primary_mass_sample.dat")
pdf_sample = np.loadtxt("frequency_sample.dat")
cdf_sample = np.loadtxt("cumulative_frequency_sample.dat")

m_min = 0.08
m_max = 60.
primary_mass_boundary = (m_min, 1.2, 3.5, 6., m_max)

########################################################################
# Interpolate the pairing function: PDF and CDF
########################################################################
pdf_interp = RegularGridInterpolator(
    (mass_ratio_sample, primary_mass_sample),
    pdf_sample.T,
    bounds_error=False,
    fill_value=0.
)
cdf_interp = RegularGridInterpolator(
    (mass_ratio_sample, primary_mass_sample),
    cdf_sample.T,
    bounds_error=False,
    fill_value=0.
)

m = np.logspace(np.log10(m_min), np.log10(60), 2**9)
m1m1, m2m2 = np.meshgrid(m, m)
z = pdf_interp((m2m2/m1m1, m1m1))
Z = cdf_interp((m2m2/m1m1, m1m1))

fig, ax = plt.subplots()
ax.pcolormesh(m, m, np.log10(z), cmap="Greys", rasterized=True)
ax.contour(m, m, np.log10(z), colors="k", levels=25)
ax.plot(m, 0.1*m, color="k", ls="solid")
ax.plot(m, 0.3*m, color="k", ls="dashed")
ax.plot(m, m, color="k", ls="solid")
ax.vlines(primary_mass_boundary, m_min, m_max, ls="dashed")
ax.set_xlim(m_min, m_max)
ax.set_ylim(m_min, m_max)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$m_{1}$")
ax.set_ylabel(r"$m_{2}$")
fig.savefig("f_M2givenM2_moe2017_pairing.pdf", dpi=300)
fig.savefig("f_M2givenM2_moe2017_pairing.jpg", dpi=300)
plt.show()

fig, ax = plt.subplots()
ax.pcolormesh(m, m, np.log10(Z), cmap="Greys", rasterized=True)
ax.contour(m, m, np.log10(Z), colors="k", levels=25)
ax.plot(m, 0.1*m, color="k", ls="solid")
ax.plot(m, 0.3*m, color="k", ls="dashed")
ax.plot(m, m, color="k", ls="solid")
ax.vlines(primary_mass_boundary, m_min, m_max, ls="dashed")
ax.set_xlim(m_min, m_max)
ax.set_ylim(m_min, m_max)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$m_{1}$")
ax.set_ylabel(r"$m_{2}$")
fig.savefig("F_M2givenM2_moe2017_pairing.pdf", dpi=300)
fig.savefig("F_M2givenM2_moe2017_pairing.jpg", dpi=300)
plt.show()

# m1 = np.linspace(m_min, m_max, 5_000)
# m2 = 1.
# za = pdf_interp((m2/m1, m1))
# m2 = 2.
# zb = pdf_interp((m2/m1, m1))
# m2 = 4.
# zc = pdf_interp((m2/m1, m1))
# m2 = 8.
# zd = pdf_interp((m2/m1, m1))
# m2 = 16.
# ze = pdf_interp((m2/m1, m1))

# fig, ax = plt.subplots()
# ax.plot(m1, za, ls="solid")
# ax.plot(m1, zb, ls="solid")
# ax.plot(m1, zc, ls="solid")
# ax.plot(m1, zd, ls="solid")
# ax.plot(m1, ze, ls="solid")
# ax.set_xlim(m_min, m_max)
# ax.set_ylim(0.1, 10.)
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlabel(r"$m_{1}$")
# # ax.set_ylabel(r"$m_{2}$")
# plt.show()
