.. _what_is_dyad:

*************
What is Dyad?
*************

Dyad is a pure-Python two-body kinematics and binary-star statistics
package for astrophysicists. It allows the user to compute the
kinematic properties of a bound gravitational two-body system given
that system's physical properties (i.e. its component masses and
orbital elements) and to synthesize a population of two-body systems
with physical properties that follow a given distribution. Dyad allows
the user to choose from a library of such distributions. This library
includes (but is not limited to):

* the distributions of binary-star mass ratios and orbital elements
  published by Duquennoy and Mayor [DM91]_ and Moe and Stefano [MS17]_
  and

* the distributions of initial stellar mass published by Kroupa [K02]_
  and Salpeter [S55]_.

For a full list of available distributions see the API documentation
for :mod:`dyad.stats`. 

The basics
==========

.. include:: ../../../dyad/__init__.py
   :start-after: -----
   :end-before: Classes

Units
=====

Dyad uses the astronomical system of units: the unit of mass is solar
mass, :math:`\mathrm{M}_{\odot}`, the unit of distance is the
astronomical unit, :math:`\mathrm{AU}`, and the unit of time is the
day, :math:`\mathrm{d}`. In this system the gravitational constant is
:math:`\text{G} = 2.959122080881949 \times
10^{-4}~\text{M}_{\odot}~\text{d}^{2}~\text{AU}^{-3}`. :numref:`table-units`
shows the units of several derived quantities and their equivalent SI values.

.. _table-units:
.. table:: The units of derived quantities in Dyad.

   +---------------------------+--------------------------------------+---------------------------------------------------------------------+
   | Quantity                  | Unit                                 | Equivalent SI value                                                 |
   +===========================+======================================+=====================================================================+
   | speed                     | :math:`\text{AU}~\text{d}^{-1}`      | :math:`1731456.8368055555~\text{m}~\text{s}^{-1}`                   |
   +---------------------------+--------------------------------------+---------------------------------------------------------------------+
   | action                    | :math:`\text{AU}^{2}~\text{d}^{-1}`  | :math:`2.590222559950685 \times 10^{17}~\text{m}^{2}~\text{s}^{-1}` |
   +---------------------------+--------------------------------------+---------------------------------------------------------------------+
   | potential                 | :math:`\text{AU}^{2}~\text{d}^{-2}`  | :math:`2997942777720.7007~\text{m}^{2}~\text{s}^{-2}`               |
   +---------------------------+--------------------------------------+---------------------------------------------------------------------+
   | specific energy           | :math:`\text{AU}^{2}~\text{d}^{-2}`  | :math:`2997942777720.7007~\text{m}^{2}~\text{s}^{-2}`               |
   +---------------------------+--------------------------------------+---------------------------------------------------------------------+
   | specific angular momentum | :math:`\text{AU}^{2}~\text{d}^{-1}`  | :math:`2.590222559950685 \times 10^{17}~\text{m}^{2}~\text{s}^{-1}` |
   +---------------------------+--------------------------------------+---------------------------------------------------------------------+

References
==========

.. [DM91]

   Duquennoy, A., and M. Mayor. 1991. \'Multiplicity among solar-type
   stars in the solar neighbourhood---II. Distribution of the orbital
   elements in an unbiased Sample\'. *Astronomy and Astrophysics* 248
   (August): 485.

.. [MS17]

    Moe, M., and R. Di Stefano. 2017. \'Mind your Ps and Qs:
    the interrelation between period (P) and mass-ratio (Q)
    distributions of binary stars.\' *The Astrophysical Journal
    Supplement Series* 230 (2): 15.
   
.. [K02]

    Kroupa, P. 2002. \'The initial mass function and its variation
    (review)\'. *ASP conference series* 285 (January): 86.

.. [S55]

    Salpeter, E. E. 1955. \'The luminosity function and stellar
    evolution.\' *The Astrophysical Journal* 121 (January): 161.
