#!/usr/bin/env python3

"""Plot mass-ratio distributions"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import dyad
import plot

mpl.style.use("sm")

#########################################################################
# Plot Moe 2017
#########################################################################
# Test utility functions: F_twin, c_twin
primary_mass_boundary = (0.8, 1.2, 3.5, 6., 60.)
log10_period_boundary = (
    0.2, 1., 1.3, 2., 2.5, 3.4, 3.5, 4., 4.5, 5.5, 6., 6.5, 8.
)

n = 50
primary_mass = np.array([1., 3.5, 7., 12.5, 25.])
log10_period = np.hstack(
    [
        np.linspace(0.2 + 1.e-6, 1., n),
        np.linspace(1., 1.3, n)[1:],
        np.linspace(1.3, 2., n)[1:],
        np.linspace(2., 2.5, n)[1:],
        np.linspace(2.5, 3.4, n)[1:],
        np.linspace(3.4, 3.5, n)[1:],
        np.linspace(3.5, 4., n)[1:],
        np.linspace(4., 4.5, n)[1:],
        np.linspace(4.5, 5.5, n)[1:],
        np.linspace(5.5, 6., n)[1:],
        np.linspace(6., 6.5, n)[1:],
        np.linspace(6.5, 8., n)[1:],
    ]
)

log10_twin_excess_period = _moe2017_log10_excess_twin_period(primary_mass)

F_twin = _moe2017_twin_excess_fraction(
    log10_period, primary_mass.reshape([-1, 1])
)
F_twin_closed_dots_x = np.array(
    [0.2, 0.2, 0.2, 0.2, 0.2, 8., 8., 8., 8., 8.]
)
F_twin_closed_dots_x += 1.e-6
F_twin_closed_dots_y = _moe2017_twin_excess_fraction(
    F_twin_closed_dots_x, np.tile(primary_mass, 2)
)

delta = _moe2017_delta(log10_period, primary_mass.reshape([-1, 1]))
c_twin = _moe2017_twin_excess_constant(
    delta, log10_period, primary_mass.reshape([-1, 1])
)
c_twin_closed_dots_x = np.array(
    [0.2, 0.2, 0.2, 0.2, 0.2, 8., 8., 8., 8., 8.]
)
c_twin_closed_dots_x += 1.e-6
c_twin_closed_dots_y = _moe2017_twin_excess_constant(
    np.hstack([np.tile(delta[0, 0], 5), np.tile(delta[0, -1], 5)]),
    c_twin_closed_dots_x,
    np.tile(primary_mass, 2)
)

fig, ax = plot.array(2, 1, sharex=True)
ax[0].plot(log10_period, F_twin[0], color="red", ls="solid",
           label=r"$M_{{1}} = {}$".format(primary_mass[0]))
ax[0].plot(log10_period, F_twin[1], color="orange", ls="solid",
           label=r"$M_{{1}} = {}$".format(primary_mass[1]))
ax[0].plot(log10_period, F_twin[2], color="green", ls="solid",
           label=r"$M_{{1}} = {}$".format(primary_mass[2]))
ax[0].plot(log10_period, F_twin[3], color="blue", ls="solid",
           label=r"$M_{{1}} = {}$".format(primary_mass[3]))
ax[0].plot(log10_period, F_twin[4], color="magenta", ls="solid",
           label=r"$M_{{1}} = {}$".format(primary_mass[4]))
ax[0].vlines(log10_twin_excess_period, -0.05, 0.45, ls="dashed")
ax[0].text(
    log10_twin_excess_period[2] + 0.15,
    0. + 0.015,
    (r"$\log_{{10}}(P_{{\text{{twin}}}}({}))$".format(primary_mass[4])
     + "\n"
     + r"$= \log_{{10}}(P_{{\text{{twin}}}}({}))$".format(primary_mass[3])
     + "\n"
     + r"$= \log_{{10}}(P_{{\text{{twin}}}}({}))$".format(primary_mass[2])
     + "\n"),
    horizontalalignment="left",
    verticalalignment="bottom",
    rotation=90.,
)
ax[0].text(
   log10_twin_excess_period[1] + 0.15,
   0. + 0.015,
   r"$\log_{{10}}(P_{{\text{{twin}}}}({}))$".format(primary_mass[1]),
   horizontalalignment="left",
   verticalalignment="bottom",
   rotation=90.,
)
ax[0].text(
   log10_twin_excess_period[0] + 0.15,
   0. + 0.015,
   r"$\log_{{10}}(P_{{\text{{twin}}}}({}))$".format(primary_mass[0]),
   horizontalalignment="left",
   verticalalignment="bottom",
   rotation=90.,
)
ax[0].scatter(F_twin_closed_dots_x[0::5], F_twin_closed_dots_y[0::5],
              s=2., color="red")
ax[0].scatter(F_twin_closed_dots_x[1::5], F_twin_closed_dots_y[1::5],
              s=2., color="orange")
ax[0].scatter(F_twin_closed_dots_x[2::5], F_twin_closed_dots_y[2::5],
              s=2., color="green")
ax[0].scatter(F_twin_closed_dots_x[3::5], F_twin_closed_dots_y[3::5],
              s=2., color="blue")
ax[0].scatter(F_twin_closed_dots_x[4::5], F_twin_closed_dots_y[4::5],
              s=2., color="magenta")
ax[0].set_xlim(-1., 9.)
ax[0].set_ylim(-0.05, 0.45)
ax[0].legend(frameon=False)
ax[0].set_ylabel(r"$F_{\text{twin}}$")
ax[1].plot(log10_period, c_twin[0], color="red", ls="solid",
           label=r"$M_{{1}} = {}$".format(primary_mass[0]))
ax[1].plot(log10_period, c_twin[1], color="orange", ls="solid",
           label=r"$M_{{1}} = {}$".format(primary_mass[1]))
ax[1].plot(log10_period, c_twin[2], color="green", ls="solid",
          label=r"$M_{{1}} = {}$".format(primary_mass[2]))
ax[1].plot(log10_period, c_twin[3], color="blue", ls="solid",
           label=r"$M_{{1}} = {}$".format(primary_mass[3]))
ax[1].plot(log10_period, c_twin[4], color="magenta", ls="solid",
           label=r"$M_{{1}} = {}$".format(primary_mass[4]))

ax[1].vlines(log10_twin_excess_period, -2., 10., ls="dashed")
ax[1].scatter(c_twin_closed_dots_x[0::5], c_twin_closed_dots_y[0::5],
              s=2., color="red")
ax[1].scatter(c_twin_closed_dots_x[1::5], c_twin_closed_dots_y[1::5],
              s=2., color="orange")
ax[1].scatter(c_twin_closed_dots_x[2::5], c_twin_closed_dots_y[2::5],
              s=2., color="green")
ax[1].scatter(c_twin_closed_dots_x[3::5], c_twin_closed_dots_y[3::5],
              s=2., color="blue")
ax[1].scatter(c_twin_closed_dots_x[4::5], c_twin_closed_dots_y[4::5],
              s=2., color="magenta")
ax[1].set_xlim(-1., 9.)
ax[1].set_ylim(-2., 10.)
ax[1].set_xlabel(r"$\log(P)$")
ax[1].set_ylabel(r"$c_{\text{twin}}$")
fig.savefig("moe2017_twin_excess.pdf")
fig.savefig("moe2017_twin_excess.jpg")
fig.show()

# Test utility functions: gamma, delta
gamma = _moe2017_gamma(log10_period, primary_mass.reshape([-1, 1]))
delta = _moe2017_delta(log10_period, primary_mass.reshape([-1, 1]))
gamma_closed_dots_x = np.array(
    [0.2, 0.2, 0.2, 0.2, 0.2, 8., 8., 8., 8., 8.]
)
gamma_closed_dots_y = _moe2017_gamma(
    gamma_closed_dots_x, np.tile(primary_mass, 2)
)
delta_closed_dots_x = np.array(
    [0.2, 0.2, 0.2, 0.2, 0.2, 8., 8., 8., 8., 8.]
)
delta_closed_dots_y = _moe2017_delta(
    delta_closed_dots_x, np.tile(primary_mass, 2)
)

fig, ax = plot.array(2, 1, sharex=True)
ax[0].plot(log10_period, delta[0], color="red", ls="solid")
ax[0].plot(log10_period, delta[1], color="orange", ls="solid")
ax[0].plot(log10_period, delta[2], color="green", ls="solid")
ax[0].plot(log10_period, delta[3], color="blue", ls="solid")
ax[0].plot(log10_period, delta[4], color="magenta", ls="solid")
ax[0].scatter(delta_closed_dots_x[0::5], delta_closed_dots_y[0::5],
              s=2., color="red")
ax[0].scatter(delta_closed_dots_x[1::5], delta_closed_dots_y[1::5],
              s=2., color="orange")
ax[0].scatter(delta_closed_dots_x[2::5], delta_closed_dots_y[2::5],
              s=2., color="green")
ax[0].scatter(delta_closed_dots_x[3::5], delta_closed_dots_y[3::5],
              s=2., color="blue")
ax[0].scatter(delta_closed_dots_x[4::5], delta_closed_dots_y[4::5],
              s=2., color="magenta")
ax[0].set_xlim(0., 8.)
ax[0].set_ylim(-3., 0.5)
ax[0].set_ylabel(r"$\delta$")
ax[1].plot(log10_period, gamma[0], color="red", ls="solid",
          label=r"$M_{{1}} = {}$".format(primary_mass[0]))
ax[1].plot(log10_period, gamma[1], color="orange", ls="solid",
          label=r"$M_{{1}} = {}$".format(primary_mass[1]))
ax[1].plot(log10_period, gamma[2], color="green", ls="solid",
          label=r"$M_{{1}} = {}$".format(primary_mass[2]))
ax[1].plot(log10_period, gamma[3], color="blue", ls="solid",
          label=r"$M_{{1}} = {}$".format(primary_mass[3]))
ax[1].plot(log10_period, gamma[4], color="magenta", ls="solid",
          label=r"$M_{{1}} = {}$".format(primary_mass[4]))
ax[1].scatter(gamma_closed_dots_x[0::5], gamma_closed_dots_y[0::5],
              s=2., color="red")
ax[1].scatter(gamma_closed_dots_x[1::5], gamma_closed_dots_y[1::5],
              s=2., color="orange")
ax[1].scatter(gamma_closed_dots_x[2::5], gamma_closed_dots_y[2::5],
              s=2., color="green")
ax[1].scatter(gamma_closed_dots_x[3::5], gamma_closed_dots_y[3::5],
              s=2., color="blue")
ax[1].scatter(gamma_closed_dots_x[4::5], gamma_closed_dots_y[4::5],
              s=2., color="magenta")
ax[1].legend(frameon=False)
ax[1].set_xlim(-1., 9.)
ax[1].set_ylim(-2.5, 1.5)
ax[1].set_xlabel(r"$\log(P)$")
ax[1].set_ylabel(r"$\gamma$")
fig.savefig("moe2017_gamma_delta.pdf")
fig.savefig("moe2017_gamma_delta.jpg")
fig.show()

# Test utility functions: twin period
primary_mass = np.linspace(0., 10., 250)
log10_period_twin = _moe2017_log10_excess_twin_period(primary_mass)

fig, ax = plot.plot()
ax.plot(primary_mass[1:], log10_period_twin[1:])
ax.scatter(0., 8., s=2., color="k", facecolor="white")
ax.set_xlim(-1., 10.)
ax.set_ylim(0., 10.)
ax.set_xlabel(r"$M_{1}$")
ax.set_ylabel(r"$\log_{10}(P_{\text{twin}})$")
fig.savefig("moe2017_twin_excess_period.pdf")
fig.savefig("moe2017_twin_excess_period.jpg")
fig.show()

# Test utility functions: norm
primary_mass_boundary = (0.8, 1.2, 3.5, 6., 60.)
log10_period_boundary = (
    0.2, 1., 1.3, 2., 2.5, 3.4, 3.5, 4., 4.5, 5.5, 6., 6.5, 8.
)

n = 50
primary_mass = np.hstack(
    [
        np.linspace(0.8, 1.2, n),
        np.linspace(1.2, 3.5, n)[1:],
        np.linspace(3.5, 6., n)[1:],
        np.linspace(6., 60., n)[1:],
    ]
)
log10_period = np.hstack(
    [
        np.linspace(0.2 + 1.e-6, 1., n),
        np.linspace(1., 1.3, n)[1:],
        np.linspace(1.3, 2., n)[1:],
        np.linspace(2., 2.5, n)[1:],
        np.linspace(2.5, 3.4, n)[1:],
        np.linspace(3.4, 3.5, n)[1:],
        np.linspace(3.5, 4., n)[1:],
        np.linspace(4., 4.5, n)[1:],
        np.linspace(4.5, 5.5, n)[1:],
        np.linspace(5.5, 6., n)[1:],
        np.linspace(6., 6.5, n)[1:],
        np.linspace(6.5, 8., n)[1:],
    ]
)

gamma = _moe2017_gamma(log10_period, primary_mass.reshape([-1, 1]))
delta = _moe2017_delta(log10_period, primary_mass.reshape([-1, 1]))
norm =_moe2017_norm(
    gamma, delta, log10_period, primary_mass.reshape([-1, 1])
)

fig, ax, cbar = plot.plot(cbar=True)
im = ax.pcolormesh(log10_period, primary_mass, norm, rasterized=True)
ax.contour(log10_period, primary_mass, norm, colors="k")
ax.vlines(log10_period_boundary, 0., 60., ls="dashed")
ax.hlines(primary_mass_boundary, 0., 8., ls="dashed")
ax.set_yscale("log")
ax.set_xlim(0.2, 8.)
ax.set_ylim(0.8, 60.)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$M_{1}$")
cbar = fig.colorbar(im, cax=cbar)
cbar.set_label(r"$A_{q}$")
plt.savefig("norm.pdf")
plt.savefig("norm.jpg")
plt.show()

# Test class methods: PDF and CDF (no twin excess)
primary_mass = np.array([1., 3.5, 7., 12.5, 25.])
print("primary_mass =", primary_mass)
# log10_excess_twin_period = _moe2017_log10_excess_twin_period(primary_mass)
# print("log10_excess_twin_period =", log10_excess_twin_period)

log10_period = 8.
rv = moe2017(log10_period, primary_mass.reshape([-1, 1]))

n = 100
q_0 = np.linspace(0., 0.1, n)[:-1]
q_1 = np.linspace(0.1, 1., n)
q_2 = np.linspace(1., 1.1, n)[1:]

pdf_0 = rv.pdf(q_0)
pdf_1 = rv.pdf(q_1)
pdf_2 = rv.pdf(q_2)

cdf_0 = rv.cdf(q_0)
cdf_1 = rv.cdf(q_1)
cdf_2 = rv.cdf(q_2)

pdf_open_dots_x = (0.1, 0.1, 0.1, 0.1, 0.1, 1., 1., 1., 1., 1.)
pdf_open_dots_y = (0., 0., 0., 0., 0., 0., 0., 0., 0., 0.)
pdf_closed_dots_x = (0.1, 0.1, 0.1, 0.1, 0.1, 1., 1., 1., 1., 1.)

pdf_closed_dots_y = (*pdf_1[:, 0], *pdf_1[:, -1])

fig, ax_1 = plot.plot()
ax_2 = ax_1.twinx()
ax_1.plot(q_0, pdf_0[0], ls="solid", color="red")
ax_1.plot(q_1, pdf_1[0], ls="solid", color="red",
          label=r"$M_{{1}} = {}$".format(primary_mass[0]))
ax_1.plot(q_2, pdf_2[0], ls="solid", color="red")
ax_1.plot(q_0, pdf_0[1], ls="solid", color="orange")
ax_1.plot(q_1, pdf_1[1], ls="solid", color="orange",
          label=r"$M_{{1}} = {}$".format(primary_mass[1]))
ax_1.plot(q_2, pdf_2[1], ls="solid", color="orange")
ax_1.plot(q_0, pdf_0[2], ls="solid", color="green")
ax_1.plot(q_1, pdf_1[2], ls="solid", color="green",
          label=r"$M_{{1}} = {}$".format(primary_mass[2]))
ax_1.plot(q_2, pdf_2[2], ls="solid", color="green")
ax_1.plot(q_0, pdf_0[3], ls="solid", color="blue")
ax_1.plot(q_1, pdf_1[3], ls="solid", color="blue",
          label=r"$M_{{1}} = {}$".format(primary_mass[3]))
ax_1.plot(q_2, pdf_2[3], ls="solid", color="blue")
ax_1.plot(q_0, pdf_0[4], ls="solid", color="magenta")
ax_1.plot(q_1, pdf_1[4], ls="solid", color="magenta",
          label=r"$M_{{1}} = {}$".format(primary_mass[4]))
ax_1.plot(q_2, pdf_2[4], ls="solid", color="magenta")
ax_1.scatter(pdf_closed_dots_x[0::5], pdf_closed_dots_y[0::5],
             s=2., color="red")
ax_1.scatter(pdf_closed_dots_x[1::5], pdf_closed_dots_y[1::5],
             s=2., color="orange")
ax_1.scatter(pdf_closed_dots_x[2::5], pdf_closed_dots_y[2::5],
             s=2., color="green")
ax_1.scatter(pdf_closed_dots_x[3::5], pdf_closed_dots_y[3::5],
             s=2., color="blue")
ax_1.scatter(pdf_closed_dots_x[4::5], pdf_closed_dots_y[4::5],
             s=2., color="magenta")
ax_1.scatter(pdf_open_dots_x[4::5], pdf_open_dots_y[4::5],
             s=2., color="magenta", facecolor="white")
ax_1.legend(frameon=False)
ax_1.set_xlabel(r"$q$")
ax_1.set_ylabel(r"$f_{q|P, M_{1}}$")
ax_1.set_ylim(-1., 11)
ax_2.plot(q_0, cdf_0[0], ls="dashed", color="red")
ax_2.plot(q_1, cdf_1[0], ls="dashed", color="red",
          label=r"$M_{{1}} = {}$".format(primary_mass[0]))
ax_2.plot(q_2, cdf_2[0], ls="dashed", color="red")
ax_2.plot(q_0, cdf_0[1], ls="dashed", color="orange")
ax_2.plot(q_1, cdf_1[1], ls="dashed", color="orange",
          label=r"$M_{{1}} = {}$".format(primary_mass[1]))
ax_2.plot(q_2, cdf_2[1], ls="dashed", color="orange")
ax_2.plot(q_0, cdf_0[2], ls="dashed", color="green")
ax_2.plot(q_1, cdf_1[2], ls="dashed", color="green",
          label=r"$M_{{1}} = {}$".format(primary_mass[2]))
ax_2.plot(q_2, cdf_2[2], ls="dashed", color="green")
ax_2.plot(q_0, cdf_0[3], ls="dashed", color="blue")
ax_2.plot(q_1, cdf_1[3], ls="dashed", color="blue",
          label=r"$M_{{1}} = {}$".format(primary_mass[3]))
ax_2.plot(q_2, cdf_2[3], ls="dashed", color="blue")
ax_2.plot(q_0, cdf_0[4], ls="dashed", color="magenta")
ax_2.plot(q_1, cdf_1[4], ls="dashed", color="magenta",
          label=r"$M_{{1}} = {}$".format(primary_mass[4]))
ax_2.plot(q_2, cdf_2[4], ls="dashed", color="magenta")
ax_2.set_ylim(-0.1, 1.1)
ax_2.set_ylabel(r"$F_{q|P, M_{1}}$")
fig.savefig("moe2017_mass_ratio_pdf.pdf")
fig.savefig("moe2017_mass_ratio_pdf.jpg")
plt.show()

# Test class methods: PPF (no twin excess)
p = np.linspace(0., 1., n)
ppf = rv.ppf(p)

fig, ax = plot.plot()
ax.plot(p, ppf[0], color="red", ls="solid",
        label=r"$M_{{1}} = {}$".format(primary_mass[0]))
ax.plot(p, ppf[1], color="orange", ls="solid",
        label=r"$M_{{1}} = {}$".format(primary_mass[1]))
ax.plot(p, ppf[2], color="green", ls="solid",
        label=r"$M_{{1}} = {}$".format(primary_mass[2]))
ax.plot(p, ppf[3], color="blue", ls="solid",
        label=r"$M_{{1}} = {}$".format(primary_mass[3]))
ax.plot(p, ppf[4], color="magenta", ls="solid",
        label=r"$M_{{1}} = {}$".format(primary_mass[4]))
ax.legend(frameon=False)
ax.set_ylim(0., 1.)
ax.set_xlabel(r"$q$")
ax.set_ylabel(r"$F^{-1}_{q|P, M_{1}}$")
fig.savefig("moe2017_mass_ratio_ppf.pdf")
fig.savefig("moe2017_mass_ratio_ppf.jpg")
plt.show()

# Test class methods: PDF and CDF (twin excess)
primary_mass = np.array([1., 3.5, 7., 12.5, 25.])
print("primary_mass =", primary_mass)
log10_excess_twin_period = _moe2017_log10_excess_twin_period(primary_mass)
print("log10_excess_twin_period =", log10_excess_twin_period)

log10_period = 0.2
# log10_period = log10_excess_twin_period + 1.
rv = moe2017(log10_period, primary_mass.reshape([-1, 1]))

n = 100
q_0 = np.linspace(0., 0.1, n)[:-1]
q_1 = np.linspace(0.1, 0.95, n)
q_2 = np.linspace(0.95, 1., n)[1:]
q_3 = np.linspace(1., 1.1, n)[1:]    

pdf_0 = rv.pdf(q_0)
pdf_1 = rv.pdf(q_1)
pdf_2 = rv.pdf(q_2)
pdf_3 = rv.pdf(q_3)

cdf_0 = rv.cdf(q_0)
cdf_1 = rv.cdf(q_1)
cdf_2 = rv.cdf(q_2)
cdf_3 = rv.cdf(q_3)

pdf_open_dots_x = (
    0.1, 0.1, 0.1, 0.1, 0.1,
    0.95, 0.95, 0.95, 0.95, 0.95,
    1., 1., 1., 1., 1.
)
pdf_open_dots_y = (
    0., 0., 0., 0., 0.,
    *pdf_2[:, 0],
    0., 0., 0., 0., 0.
)
pdf_closed_dots_x = (
    0.1, 0.1, 0.1, 0.1, 0.1,
    0.95, 0.95, 0.95, 0.95, 0.95,
    1., 1., 1., 1., 1.
)
pdf_closed_dots_y = (*pdf_1[:, 0], *pdf_1[:, -1], *pdf_2[:,-1])

fig, ax_1 = plot.plot()
ax_2 = ax_1.twinx()
ax_1.plot(q_0, pdf_0[0], ls="solid", color="red")
ax_1.plot(q_1, pdf_1[0], ls="solid", color="red",
          label=r"$M_{{1}} = {}$".format(primary_mass[0]))
ax_1.plot(q_2, pdf_2[0], ls="solid", color="red")
ax_1.plot(q_3, pdf_3[0], ls="solid", color="red")
ax_1.plot(q_0, pdf_0[1], ls="solid", color="orange")
ax_1.plot(q_1, pdf_1[1], ls="solid", color="orange",
          label=r"$M_{{1}} = {}$".format(primary_mass[1]))
ax_1.plot(q_2, pdf_2[1], ls="solid", color="orange")
ax_1.plot(q_3, pdf_3[1], ls="solid", color="orange")
ax_1.plot(q_0, pdf_0[2], ls="solid", color="green")
ax_1.plot(q_1, pdf_1[2], ls="solid", color="green",
          label=r"$M_{{1}} = {}$".format(primary_mass[2]))
ax_1.plot(q_2, pdf_2[2], ls="solid", color="green")
ax_1.plot(q_3, pdf_3[2], ls="solid", color="green")
ax_1.plot(q_0, pdf_0[3], ls="solid", color="blue")
ax_1.plot(q_1, pdf_1[3], ls="solid", color="blue",
          label=r"$M_{{1}} = {}$".format(primary_mass[3]))
ax_1.plot(q_2, pdf_2[3], ls="solid", color="blue")
ax_1.plot(q_3, pdf_3[3], ls="solid", color="blue")
ax_1.plot(q_0, pdf_0[4], ls="solid", color="magenta")
ax_1.plot(q_1, pdf_1[4], ls="solid", color="magenta",
          label=r"$M_{{1}} = {}$".format(primary_mass[4]))
ax_1.plot(q_2, pdf_2[4], ls="solid", color="magenta")
ax_1.plot(q_3, pdf_3[4], ls="solid", color="magenta")
ax_1.scatter(pdf_closed_dots_x[0::5], pdf_closed_dots_y[0::5],
             s=2., color="red")
ax_1.scatter(pdf_closed_dots_x[1::5], pdf_closed_dots_y[1::5],
             s=2., color="orange")
ax_1.scatter(pdf_closed_dots_x[2::5], pdf_closed_dots_y[2::5],
             s=2., color="green")
ax_1.scatter(pdf_closed_dots_x[3::5], pdf_closed_dots_y[3::5],
             s=2., color="blue")
ax_1.scatter(pdf_closed_dots_x[4::5], pdf_closed_dots_y[4::5],
             s=2., color="magenta")
ax_1.scatter(pdf_open_dots_x[0::5], pdf_open_dots_y[0::5],
             s=2., color="red", facecolor="white")
ax_1.scatter(pdf_open_dots_x[1::5], pdf_open_dots_y[1::5],
             s=2., color="orange", facecolor="white")
ax_1.scatter(pdf_open_dots_x[2::5], pdf_open_dots_y[2::5],
             s=2., color="green", facecolor="white")
ax_1.scatter(pdf_open_dots_x[3::5], pdf_open_dots_y[3::5],
             s=2., color="blue", facecolor="white")
ax_1.scatter(pdf_open_dots_x[4::5], pdf_open_dots_y[4::5],
             s=2., color="magenta", facecolor="white")
ax_1.legend(frameon=False, loc=2)
ax_1.set_xlabel(r"$q$")
ax_1.set_ylabel(r"$f_{q|P, M_{1}}$")
ax_1.set_ylim(-1., 11)
ax_2.plot(q_0, cdf_0[0], ls="dashed", color="red")
ax_2.plot(q_1, cdf_1[0], ls="dashed", color="red",
          label=r"$M_{{1}} = {}$".format(primary_mass[0]))
ax_2.plot(q_2, cdf_2[0], ls="dashed", color="red")
ax_2.plot(q_3, cdf_3[0], ls="dashed", color="red")
ax_2.plot(q_0, cdf_0[1], ls="dashed", color="orange")
ax_2.plot(q_1, cdf_1[1], ls="dashed", color="orange",
          label=r"$M_{{1}} = {}$".format(primary_mass[1]))
ax_2.plot(q_2, cdf_2[1], ls="dashed", color="orange")
ax_2.plot(q_3, cdf_3[1], ls="dashed", color="red")
ax_2.plot(q_0, cdf_0[2], ls="dashed", color="green")
ax_2.plot(q_1, cdf_1[2], ls="dashed", color="green",
          label=r"$M_{{1}} = {}$".format(primary_mass[2]))
ax_2.plot(q_2, cdf_2[2], ls="dashed", color="green")
ax_2.plot(q_3, cdf_3[2], ls="dashed", color="red")
ax_2.plot(q_0, cdf_0[3], ls="dashed", color="blue")
ax_2.plot(q_1, cdf_1[3], ls="dashed", color="blue",
          label=r"$M_{{1}} = {}$".format(primary_mass[3]))
ax_2.plot(q_2, cdf_2[3], ls="dashed", color="blue")
ax_2.plot(q_3, cdf_3[3], ls="dashed", color="red")
ax_2.plot(q_0, cdf_0[4], ls="dashed", color="magenta")
ax_2.plot(q_1, cdf_1[4], ls="dashed", color="magenta",
          label=r"$M_{{1}} = {}$".format(primary_mass[4]))
ax_2.plot(q_2, cdf_2[4], ls="dashed", color="magenta")
ax_2.plot(q_3, cdf_3[4], ls="dashed", color="red")
ax_2.set_ylim(-0.1, 1.1)
ax_2.set_ylabel(r"$F_{q|P, M_{1}}$")
# ax_1.set_zorder(1000.)
# ax_2.set_zorder(999.)
fig.savefig("moe2017_mass_ratio_pdf_excess.pdf")
fig.savefig("moe2017_mass_ratio_pdf_excess.jpg")
plt.show()

# Test class methods: PPF (twin excess)
p = np.linspace(0., 1., n)
ppf = rv.ppf(p)

fig, ax = plot.plot()
ax.plot(p, ppf[0], color="red", ls="solid",
        label=r"$M_{{1}} = {}$".format(primary_mass[0]))
ax.plot(p, ppf[1], color="orange", ls="solid",
        label=r"$M_{{1}} = {}$".format(primary_mass[1]))
ax.plot(p, ppf[2], color="green", ls="solid",
        label=r"$M_{{1}} = {}$".format(primary_mass[2]))
ax.plot(p, ppf[3], color="blue", ls="solid",
        label=r"$M_{{1}} = {}$".format(primary_mass[3]))
ax.plot(p, ppf[4], color="magenta", ls="solid",
        label=r"$M_{{1}} = {}$".format(primary_mass[4]))
ax.legend(frameon=False)
ax.set_ylim(0., 1.)
ax.set_xlabel(r"$p$")
ax.set_ylabel(r"$F^{-1}_{q|P, M_{1}}$")
fig.savefig("moe2017_mass_ratio_ppf_excess.pdf")
fig.savefig("moe2017_mass_ratio_ppf_excess.jpg")
plt.show()

