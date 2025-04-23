import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use("sm")

import numpy as np
from dyad.stats import mass

f_1 = 2.*(
    mass.kroupa2002.pdf(primary_mass_sample)
    *mass.kroupa2002.cdf(primary_mass_sample)
)
F_1 = cumulative_trapezoid(f_1, primary_mass_sample, initial=0.)
# f_2 = 2*(
#     mass.kroupa2002.pdf(primary_mass_sample)
#     *(1. - mass.kroupa2002.cdf(primary_mass_sample))
# )
# F_2 = cumulative_trapezoid(f_2, primary_mass_sample, initial=0.)

a = 0.8
b = 40.
primary_mass_boundary = (a, 0.5, 1.2, 3.5, 6., b)

f_M = pdf_mass(primary_mass_sample)
F_M = mass.kroupa2002().cdf(primary_mass_sample)

fig, ax = plt.subplots()
ax.plot(primary_mass_sample, f_M, ls="dashed")
ax.plot(primary_mass_sample, f_1, color="red", ls="solid")
ax.plot(primary_mass_sample, frequency_sample, ls="solid")
ax.vlines(primary_mass_boundary, 1.e-5, 1.e1, ls="dotted")
ax.set_xlim(a, b)
ax.set_ylim(1.e-5, 1.e1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$m/\text{M}_{\odot}$")
# ax.set_ylabel(r"$f_{M_{1}}$")
fig.savefig("./Figures/f_M_moe2017_pairing.jpg")
fig.savefig("./Figures/f_M_moe2017_pairing.pdf")
plt.show()

fig, ax = plt.subplots()
ax.plot(primary_mass_sample, F_1, ls="solid")
ax.plot(primary_mass_sample, F_M, ls="dashed")
ax.plot(primary_mass_sample, cumulative_frequency_sample, ls="solid")
ax.vlines(primary_mass_boundary, 0., 1.2, ls="dotted")
ax.set_xlim(a, b)
ax.set_ylim(0., 1.2)
ax.set_xscale("log")
ax.set_xlabel(r"$m/\text{M}_{\odot}$")
# ax.set_ylabel(r"$F_{M_{1}}$")
fig.savefig("./Figures/f_M_moe2017_pairing.jpg")
fig.savefig("./Figures/f_M_moe2017_pairing.pdf")
plt.show()
