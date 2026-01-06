"""
.. dyad_api:

=========
Constants
=========

.. currentmodule:: dyad._constants

This module defines the physical constants used by Dyad. It uses the
values define by :mod:`scipy._constants` where they are available and
defines new values where they are not.

"""

from scipy.constants import (au, day, gravitational_constant)

#: Solar mass (:math:`\mathrm{kg}`)
M_SUN = 1.988_409_87e+30
#: Astronomical unit (:math:`\mathrm{m}`)
AU = au
#: Day (:math:`\mathrm{s}`)
DAY = day
#: Gravitational constant
#: (:math:`\mathrm{au}~\mathrm{day}^{-2}~\mathrm{M}_{\odot}^{-1}`)
GRAV_CONST = gravitational_constant*M_SUN*DAY**2./AU**3.
