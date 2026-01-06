#!/usr/bin/env python3

"""
Code for tutorial_3.rst.

"""

# import plot

import numpy as np
m_1 = np.array([1., 3.5, 7., 12.5, 28.])

import dyad.stats as stats
log_p = np.linspace(-1., 9., 500)
f = stats.log_period.moe2017(m_1[:,None]).pdf(log_p)

import matplotlib.pyplot as plt
label = [
    r"$M_{1} = 1\mathrm{M}_{\odot}$",
    r"$M_{1} = 3.5\mathrm{M}_{\odot}$",
    r"$M_{1} = 7\mathrm{M}_{\odot}$",
    r"$M_{1} = 12.5\mathrm{M}_{\odot}$",
    r"$M_{1} = 28\mathrm{M}_{\odot}$"
]
color = ["red", "orange", "green", "blue", "magenta"]
# fig, ax = plot.plot()
fig, ax = plt.subplots()
for (f_i, label_i, color_i) in zip(f, label, color):
    ax.plot(log_p, f_i, label=label_i, color=color_i)
ax.legend(frameon=False)
ax.set_ylim(0., 0.3)
ax.set_xlabel(r"$\log_{10}(p/\mathrm{d})$")
ax.set_ylabel(r"$f_{\log_{10}(P)}$")
fig.savefig("./moe2017_pdf_logperiod.pdf", dpi=300)
fig.savefig("./moe2017_pdf_logperiod.jpg", dpi=300)
plt.show()

p = np.logspace(-1., 9.)
f = stats.period.moe2017(m_1[:,None]).pdf(p)

q = np.linspace(0., 1.1, 500)
f = stats.mass_ratio.moe2017(0.2, m_1[:,None]).pdf(q)

# fig, ax = plot.plot()
fig, ax = plt.subplots()
for (f_i, label_i, color_i) in zip(f, label, color):
    ax.plot(q, f_i, label=label_i, color=color_i)
ax.legend(frameon=False)
ax.set_xlabel(r"$q$")
ax.set_ylabel(r"$f_Q$")
fig.savefig("./moe2017_pdf_mass_ratio_short_period.pdf", dpi=300)
fig.savefig("./moe2017_pdf_mass_ratio_short_period.jpg", dpi=300)
plt.show()

f = stats.mass_ratio.moe2017(8., m_1[:,None]).pdf(q)

# fig, ax = plot.plot()
fig, ax = plt.subplots()
for (f_i, label_i, color_i) in zip(f, label, color):
    ax.plot(q, f_i, label=label_i, color=color_i)
ax.legend(frameon=False)
ax.set_xlabel(r"$q$")
ax.set_ylabel(r"$f_Q$")
fig.savefig("./moe2017_pdf_mass_ratio_long_period.pdf", dpi=300)
fig.savefig("./moe2017_pdf_mass_ratio_long_period.jpg", dpi=300)
plt.show()

e = np.linspace(-0.1, 1.1, 500)
f = stats.eccentricity.moe2017(1., m_1[:,None]).pdf(e)

# fig, ax = plot.plot()
fig, ax = plt.subplots()
for (f_i, label_i, color_i) in zip(f, label, color):
    ax.plot(e, f_i, label=label_i, color=color_i)
ax.legend(frameon=False)
ax.set_ylim(0., 5.)
ax.set_xlabel(r"$e$")
ax.set_ylabel(r"$f_E$")
fig.savefig("./moe2017_pdf_eccentricity_short_period.pdf", dpi=300)
fig.savefig("./moe2017_pdf_eccentricity_short_period.jpg", dpi=300)
plt.show()

f = stats.eccentricity.moe2017(8., m_1[:,None]).pdf(e)

# fig, ax = plot.plot()
fig, ax = plt.subplots()
for (f_i, label_i, color_i) in zip(f, label, color):
    ax.plot(e, f_i, label=label_i, color=color_i)
ax.legend(frameon=False)
ax.set_xlabel(r"$e$")
ax.set_ylabel(r"$f_E$")
fig.savefig("./moe2017_pdf_eccentricity_long_period.pdf", dpi=300)
fig.savefig("./moe2017_pdf_eccentricity_long_period.jpg", dpi=300)
plt.show()

n_binary = 10_000
m_1 = stats.mass.salpeter1955(0.8, 40.).rvs(size=n_binary)
log_p = stats.log_period.moe2017(m_1).rvs()
p = 10.**log_p
q = stats.mass_ratio.moe2017(log_p, m_1).rvs()
e = np.zeros(n_binary)
idx = (log_p > 0.9375)
e[idx] = stats.eccentricity.moe2017(log_p[idx], m_1[idx]).rvs()
Omega = stats.longitude_of_ascending_node.rvs(size=n_binary)
i = stats.inclination.rvs(size=n_binary)
omega = stats.argument_of_pericentre.rvs(size=n_binary)
import dyad
a = dyad.semimajor_axis_from_period(p, m_1, m_1*q)
a_1 = dyad.primary_semimajor_axis_from_semimajor_axis(a, q)
binary = dyad.TwoBody(m_1, q, a_1, e, Omega, i, omega)
theta = stats.true_anomaly(e).rvs()
r_1 = binary.primary.radius(theta)
r_2 = binary.secondary.radius(theta)
bins = np.logspace(-3., 6., 25)
edge_r1, count_r1 = np.histogram(r_1, bins=bins)
edge_r2, count_r2 = np.histogram(r_2, bins=bins)

# fig, ax = plot.plot()
fig, ax = plt.subplots()
# ax.stairs(edge_r1, count_r1, label="primary")
# ax.stairs(edge_r2, count_r2, label="primary")
ax.hist(np.log10(r_1), bins="auto", alpha=0.2)
ax.hist(np.log10(r_2), bins="auto", alpha=0.2)
ax.legend(frameon=False)
# ax.set_xscale("log")
ax.set_xlabel(r"$\log_{10}(r/\mathrm{AU})$")
ax.set_ylabel(r"$\nu$")
fig.savefig("./moe2017_sample_radius.pdf", dpi=300)
fig.savefig("./moe2017_sample_radius.jpg", dpi=300)
plt.show()

v_1 = binary.primary.speed(theta)
v_2 = binary.secondary.speed(theta)
bins = np.logspace(-3., 6., 25)
edge_v1, count_v1 = np.histogram(r_1, bins=bins)
edge_v2, count_v2 = np.histogram(r_2, bins=bins)

# fig, ax = plot.plot()
fig, ax = plt.subplots()
# ax.stairs(edge_v1, count_v1, label="primary")
# ax.stairs(edge_v2, count_v2, label="primary")
ax.hist(np.log10(v_1), bins="auto", alpha=0.2)
ax.hist(np.log10(v_2), bins="auto", alpha=0.2)
ax.legend(frameon=False)
# ax.set_xscale("log")
ax.set_xlabel(r"$\log_{10}(v/\mathrm{AU}~\mathrm{d}^{-1})$")
ax.set_ylabel(r"$\nu$")
fig.savefig("./moe2017_sample_speed.pdf", dpi=300)
fig.savefig("./moe2017_sample_speed.jpg", dpi=300)
plt.show()
