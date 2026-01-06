#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# import plot

from elliptical_orbit import AngleAnnotation

# mpl.style.use("sm")

from matplotlib.patches import Arc
from matplotlib.transforms import Bbox, IdentityTransform, TransformedBbox

if __name__ == "__main__":
    m_1 = 1.
    m_2 = 0.75
    theta = 7.*np.pi/8.
    e = 0.75
    a_1 = 1.
    b_1 = a_1*np.sqrt(1. - e**2.)
    a_2 = a_1*m_2/m_1
    b_2 = a_2*np.sqrt(1. - e**2.)
    r_1 = a_1*(1. - e**2.)/(1 + e*np.cos(theta))
    x_1 = r_1*np.cos(theta) + a_1*e
    y_1 = r_1*np.sin(theta)
    x_2 = -x_1*m_2/m_1
    y_2 = -y_1*m_2/m_1

    fig, ax = plt.subplots()
    # Plot primary ellipse
    ellipse = mpl.patches.Ellipse(
        xy=(-a_1*e, 0.), width=2.*a_1, height=2.*b_1, edgecolor="k", fc="None"
    )
    ax.add_patch(ellipse)
    ax.scatter(0., 0., color="k")
    # Plot secondary ellipse
    ellipse = mpl.patches.Ellipse(
        xy=(a_2*e, 0.), width=2.*a_2, height=2.*b_2, edgecolor="k", fc="None"
    )
    ax.add_patch(ellipse)
    # Plot bodies
    ax.scatter(x_1 - a_1*e, y_1, color="k")
    ax.scatter(x_2 + a_2*e, y_2, color="k")
    ax.arrow(0., 0., x_1 - a_1*e, y_1, shape='full', length_includes_head=True,
              head_width=0.05, color="k")
    ax.arrow(0., 0., x_2 + a_2*e, y_2, shape='full', length_includes_head=True,
              head_width=0.05, color="k")
    ax.annotate("$\mathbf{r}_{1}$", xy=(x_1 - a_1*e, y_1), xytext=(-2., 2.),
                textcoords="offset points", va="bottom", ha="center")
    ax.annotate("$\mathbf{r}_{2}$", xy=(x_2 + a_2*e, y_2), xytext=(2., -2.),
                textcoords="offset points", va="top", ha="left")
    AngleAnnotation((0., 0.), (a_1, 0.), (x_1 - a_1*e, y_1), ax=ax, size=40.,
                    text=r"$\theta$", textposition="outside")
    # Plot furniture
    ax.set_xticks([-2., -1., 0., 1., 2.])
    ax.set_yticks([-1., 0., 1.])
    ax.set_xlim(-2.*a_1, 1.5*a_1)
    ax.set_ylim(-1.75*a_1/np.sqrt(2.), 1.75*a_1/np.sqrt(2.))
    ax.vlines((0.,), *ax.get_ylim(), ls="dotted", zorder=0.)
    ax.hlines((0.,), *ax.get_xlim(), ls="dotted", zorder=0.)
    ax.set_xlabel(r"$p/a_{1}$")
    ax.set_ylabel(r"$q/a_{1}$")
    # Save figures
    fig.savefig("./twobody_system.pdf")
    fig.savefig("./twobody_system.jpg")
    plt.show()
