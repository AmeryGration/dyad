.. _what_is_dyad:

*************
What is Dyad?
*************

Dyad is a pure-Python binary star kinematics and statistics package for astrophysicists. It allows the user to compute the kinematic properties of a bound gravitational two-body system given that system's physical properties (i.e. its component masses and orbital elements) and to synthesize a population of two-body systems with physical properties that follow a given distribution. Dyad allows the user to choose from a library of such distributions. This library includes (but is not limited to):

* the distributions published by Duquennoy and Mayor [DM91]_ and Moe and Stefano [MS17]_ for the mass-ratios and orbital elements of binary stars in the Solar neighbourhood, and

* the distributions published by Chabrier [C03]_, Kroupa [K02]_, and Salpeter [S55]_ for the initial stellar mass.

For a full list of available distributions see the API documentation for :mod:`dyad.stats`. Dyad also allows the user to compute the distributions of the primary- and secondary-star masses in a manner consistent with a given mass function and pairing function. It does so by implementing the method of Gration, Izzard, and Das [GID25]_.

The basics
==========

.. include:: ../../../dyad/__init__.py
   :start-after: `\mathrm{M}_{\odot}`.
   :end-before: Classes

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

.. [C03]

   Chabrier, G. 2003. \'Galactic stellar and substellar initial
   mass function\'. *Publications of the Astronomical Society of the
   Pacific* 115 (July): 763--95. 
   
.. [K02]

    Kroupa, P. 2002. \'The initial mass function and its variation
    (review)\'. *ASP conference series* 285 (January): 86.

.. [S55]

    Salpeter, E. E. 1955. \'The luminosity function and stellar
    evolution.\' *The Astrophysical Journal* 121 (January): 161.

.. [GID25]

   Gration, A., Izzard, R., and P. Das. Forthcoming.
