#!/usr/bin/env python

"""Plot Moe 2017"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from distributions import *

mpl.style.use("sm")

primary_mass = np.array([1., 3.5, 7., 12.5, 25.])
rv = moe2017(primary_mass.reshape([-1, 1]))

n = 500
x_1 = np.linspace(-1., 0.2, n)[:-1]
x_2 = np.linspace(0.2, 8., n)
x_3 = np.linspace(8., 9., n)[1:]

pdf_1 = rv.pdf(x_1)
pdf_2 = rv.pdf(x_2)
pdf_3 = rv.pdf(x_3)

cdf_1 = rv.cdf(x_1)
cdf_2 = rv.cdf(x_2)
cdf_3 = rv.cdf(x_3)

pdf_open_dots_x = (0.2, 0.2, 0.2, 0.2, 0.2, 8., 8., 8., 8., 8.)
pdf_open_dots_y = (0., 0., 0., 0., 0., 0., 0., 0., 0., 0.)
pdf_closed_dots_x = (0.2, 0.2, 0.2, 0.2, 0.2, 8., 8., 8., 8., 8.)
pdf_closed_dots_y = (*pdf_2[:, 0], *pdf_2[:, -1])

fig, ax_1 = plt.subplots()
ax_2 = ax_1.twinx()
ax_1.plot(x_1, pdf_1[0], color="red", ls="solid" )
ax_1.plot(x_2, pdf_2[0], color="red", ls="solid",
          label=r"$M_{{1}} = {}$".format(primary_mass[0]))
ax_1.plot(x_3, pdf_3[0], color="red", ls="solid")
ax_1.plot(x_1, pdf_1[1], color="orange", ls="solid" )
ax_1.plot(x_2, pdf_2[1], color="orange", ls="solid",
          label=r"$M_{{1}} = {}$".format(primary_mass[1]))
ax_1.plot(x_3, pdf_3[1], color="orange", ls="solid")
ax_1.plot(x_1, pdf_1[2], color="green", ls="solid" )
ax_1.plot(x_2, pdf_2[2], color="green", ls="solid",
          label=r"$M_{{1}} = {}$".format(primary_mass[2]))
ax_1.plot(x_3, pdf_3[2], color="green", ls="solid")
ax_1.plot(x_1, pdf_1[3], color="blue", ls="solid" )
ax_1.plot(x_2, pdf_2[3], color="blue", ls="solid",
          label=r"$M_{{1}} = {}$".format(primary_mass[3]))
ax_1.plot(x_3, pdf_3[3], color="blue", ls="solid")
ax_1.plot(x_1, pdf_1[4], color="magenta", ls="solid" )
ax_1.plot(x_2, pdf_2[4], color="magenta", ls="solid",
          label=r"$M_{{1}} = {}$".format(primary_mass[4]))
ax_1.plot(x_3, pdf_3[4], color="magenta", ls="solid")
ax_1.legend(frameon=False, loc=2)
ax_1.set_xlim(-1., 9.)
ax_1.set_ylim(-0.05, 0.25)
ax_1.set_xlabel(r"$x$")
ax_1.set_ylabel(r"$f_{X|M_{1}}$")
ax_2.plot(x_1, cdf_1[0], color="red", ls="dashed" )
ax_2.plot(x_2, cdf_2[0], color="red", ls="dashed",
          label=r"$M_{{1}} = {}$".format(primary_mass[0]))
ax_2.plot(x_3, cdf_3[0], color="red", ls="dashed")
ax_2.plot(x_1, cdf_1[1], color="orange", ls="dashed" )
ax_2.plot(x_2, cdf_2[1], color="orange", ls="dashed",
          label=r"$M_{{1}} = {}$".format(primary_mass[1]))
ax_2.plot(x_3, cdf_3[1], color="orange", ls="dashed")
ax_2.plot(x_1, cdf_1[2], color="green", ls="dashed" )
ax_2.plot(x_2, cdf_2[2], color="green", ls="dashed",
          label=r"$M_{{1}} = {}$".format(primary_mass[2]))
ax_2.plot(x_3, cdf_3[2], color="green", ls="dashed")
ax_2.plot(x_1, cdf_1[3], color="blue", ls="dashed" )
ax_2.plot(x_2, cdf_2[3], color="blue", ls="dashed",
          label=r"$M_{{1}} = {}$".format(primary_mass[3]))
ax_2.plot(x_3, cdf_3[3], color="blue", ls="dashed")
ax_2.plot(x_1, cdf_1[4], color="magenta", ls="dashed" )
ax_2.plot(x_2, cdf_2[4], color="magenta", ls="dashed",
          label=r"$M_{{1}} = {}$".format(primary_mass[4]))
ax_2.plot(x_3, cdf_3[4], color="magenta", ls="dashed")
ax_2.set_ylim(-0.25, 1.25)
ax_2.set_ylabel(r"$F_{X|M_{1}}$")

ax_1.scatter(pdf_closed_dots_x[0::5], pdf_closed_dots_y[0::5],
             s=2., color="red", zorder=np.inf)
ax_1.scatter(pdf_closed_dots_x[1::5], pdf_closed_dots_y[1::5],
             s=2., color="orange", zorder=np.inf)
ax_1.scatter(pdf_closed_dots_x[2::5], pdf_closed_dots_y[2::5],
             s=2., color="green", zorder=np.inf)
ax_1.scatter(pdf_closed_dots_x[3::5], pdf_closed_dots_y[3::5],
             s=2., color="blue", zorder=np.inf)
ax_1.scatter(pdf_closed_dots_x[4::5], pdf_closed_dots_y[4::5],
             s=2., color="magenta", zorder=np.inf)
ax_1.scatter(pdf_open_dots_x[4::5], pdf_open_dots_y[4::5],
             s=2., color="magenta", facecolor="white", zorder=np.inf)

# plt.savefig("moe2017_logperiod_pdf.pdf")
# plt.savefig("moe2017_logperiod_pdf.jpg")
plt.show()

# Test class methods: PPF (no twin excess)
p = np.linspace(0., 1., 50)
ppf = rv.ppf(p)

fig, ax = plt.subplots()
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
ax.legend(frameon=False, loc=2)
ax.set_ylim(0., 8.)
ax.set_xlabel(r"$p$")
ax.set_ylabel(r"$F^{-1}_{q|P, M_{1}}$")
# fig.savefig("moe2017_logperiod_ppf.pdf")
# fig.savefig("moe2017_logperiod_ppf.jpg")
plt.show()
