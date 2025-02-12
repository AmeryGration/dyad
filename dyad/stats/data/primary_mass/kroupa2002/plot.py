import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use("sm")

import numpy as np
from dyad.stats import mass

f = 2.*(
    mass.kroupa2002.pdf(primary_mass_sample)
    *mass.kroupa2002.cdf(primary_mass_sample)
)
F = cumulative_trapezoid(f, primary_mass_sample, initial=0.)

a = 0.08
b = 60.
primary_mass_boundary = (a, 0.5, 1.2, 3.5, 6., b)

imf = pdf_mass(primary_mass_sample)

fig, ax = plt.subplots()
ax.plot(primary_mass_sample, imf, ls="dashed")
ax.plot(primary_mass_sample, f, color="red", ls="solid")
# ax.plot(primary_mass_sample, F, color="red")
ax.plot(primary_mass_sample, frequency_sample, ls="solid")
# ax.plot(primary_mass_sample, cumulative_frequency_sample, ls="solid")
ax.vlines(primary_mass_boundary, 1.e-5, 1.e1, ls="dotted")
ax.set_xlim(a, b)
ax.set_ylim(1.e-5, 1.e1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$m/\text{M}_{\odot}$")
ax.set_ylabel(r"$f_{M_{1}}$")
fig.savefig("./Figures/f_M_moe2017_pairing.jpg")
fig.savefig("./Figures/f_M_moe2017_pairing.pdf")
plt.show()
