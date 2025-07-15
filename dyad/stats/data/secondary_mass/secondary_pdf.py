#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.integrate import trapezoid
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator
from dyad.stats import mass_ratio
from dyad.stats import period as period

mpl.style.use("sm")

########################################################################
# Interpolate the pairing function: PDF and CDF
########################################################################
pdf_interp = RegularGridInterpolator(
    (mass_ratio_sample, primary_mass_sample),
    pdf_sample.T,
    bounds_error=False,
    fill_value=0.
)
# cdf_interp = RegularGridInterpolator(
#     (mass_ratio_sample, primary_mass_sample),
#     cdf_sample.T,
#     bounds_error=False,
#     fill_value=0.
# )

########################################################################
#
########################################################################
# m = np.linspace(0.8, 40., 2**9)
m_1 = np.logspace(np.log10(0.8), np.log10(40.), 2**9)
m_2 = np.logspace(np.log10(0.8), np.log10(40.), 2**9)
m1m1, m2m2 = np.meshgrid(m_1, m_2)
z = pdf_interp((m2m2/m1m1, m1m1))/m1m1
Z = cumulative_trapezoid(z, m_2, axis=0, initial=0.)
z /= Z[-1]
Z /= Z[-1]

########################################################################
# Plot PDF
########################################################################
fig, ax = plt.subplots()
ax.pcolormesh(m1m1, m2m2, np.log10(z), rasterized=True)#, cmap="Greys")
ax.contour(m1m1, m2m2, np.log10(z), colors="k")
ax.plot(m_1, 0.1*m_1, color="k", ls="solid")
ax.plot(m_1, 0.3*m_1, color="k", ls="dashed")
ax.plot(m_1, 0.95*m_1, color="k", ls="dashed")
# ax.plot(m_1, m_2, color="k", ls="solid")
ax.vlines(primary_mass_boundary, 0.8, 40., ls="dashed")
ax.set_xlim(0.8, 40.)
ax.set_ylim(0.8, 40.)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$m_{1}$")
ax.set_ylabel(r"$m_{2}$")
ax.set_title(r"$f_{M_{2}|M_{1}}$")
fig.savefig("f_M2givenM2_moe2017_pairing.pdf", dpi=300)
fig.savefig("f_M2givenM2_moe2017_pairing.jpg", dpi=300)
plt.show()

########################################################################
# Plot PDF
########################################################################
fig, ax_1 = plt.subplots()
ax_2 = ax_1.twinx()
ax_1.pcolormesh(m1m1, m2m2, Z, rasterized=True)#, cmap="Greys")
ax_1.contour(m1m1, m2m2, Z, colors="k")
ax_1.plot(m_1, 0.1*m_1, color="k", ls="solid")
ax_1.plot(m_1, 0.3*m_1, color="k", ls="dashed")
ax_1.plot(m_1, 0.95*m_1, color="k", ls="dashed")
ax_1.plot(m_1, m_1, color="k", ls="solid")
ax_1.vlines(primary_mass_boundary, 0.8, 40., ls="dashed")
ax_1.set_xlim(0.8, 40.)
ax_1.set_ylim(0.8, 40.)
ax_1.set_xscale("log")
ax_1.set_yscale("log")
ax_1.set_xlabel(r"$m_{1}$")
ax_1.set_ylabel(r"$m_{2}$")
ax_1.set_title(r"$f_{M_{2}|M_{1}}$")
ax_2.plot(m_1, Z[-1], color="red")
ax_2.set_ylim(0., 1.2)
ax_2.set_ylabel(r"$I(m_{1})$")
fig.savefig("F_M2givenM2_moe2017_pairing.pdf", dpi=300)
fig.savefig("F_M2givenM2_moe2017_pairing.jpg", dpi=300)
plt.show()

########################################################################
# 
########################################################################
# m_1 = np.array([0.85, 1., 14., 27., 40.])[:,None]
# m_1 = np.logspace(np.log10(0.85), np.log10(40), 5)[:,None]
# m_2 = np.logspace(np.log10(0.8), np.log10(40.), 5_000)
# z = pdf_interp((m_2/m_1, m_1))/m_1

label = [r"$m_1 = {}$".format(m) for m in m_1.flatten()]
color = [a*np.ones(3) for a in np.linspace(0., 0.8, 5)]

########################################################################
# Plot PDF for given m_1
########################################################################
fig, ax = plt.subplots()
for z_i, label_i, color_i in zip(z[50::50], label, color):
    ax.plot(m_2, z_i, color=color_i, label=label_i, ls="solid")
# ax.set_xlim(0.8, 40.)
# ax.set_ylim(0.1, 10.)
ax.legend(frameon=False, loc=1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$m_{2}$")
ax.set_ylabel(r"$f_{M_{2}|M_{1}}$")

plt.show()

# ########################################################################
# # Define the conditional PDF for secondary mass given primary mass_ratio
# ########################################################################
# from scipy.integrate import quad
# from scipy.integrate import dblquad

# # def f_M2givenM1(m_2, m_1, q):

# def f_QPgivenM(p, q, m_1):
#     """Return the joint probability density for given mass ratio and period"""
#     res = mass_ratio.moe2017(np.log10(p), m_1).pdf(q)*period.moe2017(m).pdf(p)

#     return res

# def f_QgivenM(q, m_1):
#     res = quad(f_QPgivenM, p_min, p_max, args=(q, m_1,),
#                points=period_boundary)

#     return res

# p_min = 10.**0.2
# p_max = 1.e+8
# q = 0.1
# m_1 = 5.
# log10_period_boundary = (
#     0.2, 1., 1.3, 2., 2.5, 3.4, 3.5, 4., 4.5, 5.5, 6., 6.5, 8.
# )
# period_boundary = 10.**np.array(log10_period_boundary)

# res = f_QgivenM(q, m_1)

# # f_ = quad(g, p_min, p_max, args=(q, m_1,))
# # print(I)

# n = 1_000
# m_1 = 0.85
# # m_2 = np.hstack([
# #     np.logspace(np.log10(0.8), np.log10(0.95*m_1), n),
# #     np.logspace(np.log10(0.95*m_1), np.log10(1.), n)[1:],
# # ])
# m_2 = np.hstack([
#     np.logspace(np.log10(0.8), np.log10(0.95*m_1), n)[:-1],
#     np.linspace(0.95*m_1, m_1, n),
# ])
# z = pdf_interp((m_2/m_1, m_1))/m_1
# Z = cumulative_trapezoid(z, m_2, axis=0, initial=0.)
# print(Z[-1])

# plt.plot(m_2, z)
# plt.plot(m_2, Z)
# plt.show()

