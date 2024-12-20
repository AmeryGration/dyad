.. _tutorial_2:

*****************************************
Synthesizing a population of binary stars
*****************************************

Dyad has a subpackage, :mod:`dyad.stats`, that contains probability distributions for the masses, mass ratios, and orbital elements of a population of binary stars. These probability distribitions are implemented in the same way as Scipy's continous random variables (see here and, for an example, here). As a result, they come equiped with a large number of useful methods, in particular ``pdf``, which computes the probability density function (PDF), and ``rvs``, which generates a sample.

For example, consider the random variable for inclination, :math:`I`, where :math:`\sin(I) \sim U(0, \pi)`. This is implemented by Dyad using the class :class:`dyad.stats.inclination`. Instantiate this class as follows.

.. sourcecode:: python

   >>> import dyad.stats as stats
   >>> rv = stats.inclination

Now compute the PDF on the interval :math:`[0, \pi]`.

.. sourcecode:: python

   >>> import numpy as np
   >>> x = np.linspace(0., np.pi)
   >>> pdf = rv.pdf(x)

And synthesize :math:`10~000` realizations.

.. sourcecode:: python

   >>> sample = rv.rvs(size=10_000)

Now plot our results.

.. sourcecode:: python

   >>> import matplot.pyplot as plt
   >>> fig, ax = plt.subplots()
   >>> ax.histogram(sample, bins=25, density=True, histtype="step")
   >>> ax.plot(x, pdf)
   >>> plt.show()

.. image

Some of Dyad's random variables are conditional on other random variables. We must use their shape parameters to fully specify them. For example, the true anomaly of a body moving on an elliptical orbit in a gravitational central potentia depends on that orbit's eccentricity, and so is a conditional random variable, :math:`\Theta|E = e`. Again, we may synthesize a sample and compute the PDF. Suppose that :math:`e = 0.5`.

.. sourcecode:: python

   >>> rv = stats.true_anomaly(e=0.5)
   >>> x = np.linspace(0., 2.*np.pi)
   >>> pdf = rv.pdf(x)
   >>> sample = rv.rvs(size=10_000)

Plot the results.

.. sourcecode:: python

   >>> fig, ax = plt.subplots()
   >>> ax.histogram(sample, bins=25, density=True, histtype="step")
   >>> ax.plot(x, pdf)
   >>> plt.show()

In some cases there is a choice of distribution. These are kept in the
submodules :mod:`dyad.stats.eccentricity`, :mod:`dyad.stats.period`,
:mod:`dyad.stats.log_period`, :mod:`dyad.stats.mass`,
:mod:`dyad.stats.mass_ratio`, :mod:`dyad.stats.semimajor_axis`. For
example, when considering the eccentricity of an orbit we may wish to
use a thermal distribution.

.. sourcecode:: python

   >>> rv = stats.eccentricity.thermal

Its methods are available in the same way as before.

.. sourcecode:: python

   >>> x = np.linspace(0., 1.)
   >>> rv.pdf(x)
   array([0.        , 0.04081633, 0.08163265, 0.12244898, 0.16326531,
	  0.20408163, 0.24489796, 0.28571429, 0.32653061, 0.36734694,
	  0.40816327, 0.44897959, 0.48979592, 0.53061224, 0.57142857,
	  0.6122449 , 0.65306122, 0.69387755, 0.73469388, 0.7755102 ,
	  0.81632653, 0.85714286, 0.89795918, 0.93877551, 0.97959184,
	  1.02040816, 1.06122449, 1.10204082, 1.14285714, 1.18367347,
	  1.2244898 , 1.26530612, 1.30612245, 1.34693878, 1.3877551 ,
	  1.42857143, 1.46938776, 1.51020408, 1.55102041, 1.59183673,
	  1.63265306, 1.67346939, 1.71428571, 1.75510204, 1.79591837,
	  1.83673469, 1.87755102, 1.91836735, 1.95918367, 2.        ])
   
A complete population
=====================

Let us synthesize the complete orbital properties of a population of binary stars: mass, mass ratio, and orbital elements. We will use the distributions of Duquennoy and Mayor [DM91]_. Assume that the primary stars of our populations have masses of :math:`0.8~\mathrm{M}_{\odot}` and sample the mass ratio and the period.

.. sourcecode:: python

   >>> n = 10_000
   >>> m_1 = np.full((n,), 0.8)
   >>> q = stats.mass_ratio.duquennoy1991.rvs(size=n)
   >>> p = stats.period.duquennoy1991.rvs(size=n)

Now sample the eccentricity, remembering that the circularization period is :math:`11~\mathrm{day}`. 

.. sourcecode:: python
		
   >>> e = np.zeros(n)
   >>> e[p > 11.] = stats.eccentricity.duquennoy1991(p[p > 11.]).rvs(size=n)

Using these eccentricities sample the true anomaly.

.. sourcecode:: python

   >>> theta = stats.true_anomaly(e).rvs()

Note that, since the eccentricities are all different, we do not pass a size argument to the method ``rvs``. Now sample the orientation of the system.

.. sourcecode:: python

   >>> Omega = stats.longitude_of_ascending_node.rvs(size=n)
   >>> i = stats.inclination.rvs(size=n)
   >>> omega = stats.argument_of_pericentre().rvs(size=n)

The class :class:`dyad.TwoBody` can serve as a container for these values. First convert the periods to their equivalent primary-star semimajor axes.

.. sourcecode:: python

   >>> a = dyad.semimajor_axis_from_period(p, m_1, m_1*q)
   >>> a_1 = dyad.primary_semimajor_axis_from_semimajor_axis(a)

Then instantiate a :class:`dyad.TwoBody` object.

.. sourcecode:: python

   >>> binary = dyad.TwoBody(m_1, q, a_1, e, theta, Omega, i, omega)

Inspect the phase state of the 42nd element, which is given in Cartesian coordinates as :math:`(x, y, z, v_{x}, v_{y}, v_{z})`.

.. sourcecode:: python

   >>> binary.primary.state[42]

References
==========

.. [DM91]

   Duquennoy, A., and M. Mayor. 1991. \'Multiplicity among solar-type
   stars in the solar neighbourhood---II. Distribution of the orbital
   elements in an unbiased Sample\'. *Astronomy and Astrophysics* 248
   (August): 485.
