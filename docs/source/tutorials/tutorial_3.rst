.. _tutorial_3:

********************************************
The distributions of Moe & Di Stefano (2017)
********************************************

Moe and Di Stefano [MS17]_ reported empirical distributions for the period, :math:`P`, mass-ratio, :math:`Q`, and eccentricity, :math:`E`, of binary stars in the solar neighbourhood.
Their catalogue included data for binary systems with primary stars of five spectral types, namely

- solar, with :math:`m_{1}/\mathrm{M}_{\odot} \in [0.8, 1.2)`,
- A and late-B, with :math:`m_{1}/\mathrm{M}_{\odot} \in [2, 5)`,
- mid-B, with :math:`m_{1}/\mathrm{M}_{\odot} \in [5, 9)`,
- early B, with :math:`m_{1}/\mathrm{M}_{\odot} \in [9, 16)`, and
- O, with :math:`m_{1}/\mathrm{M}_{\odot} \in [16, 40]`,
  
where :math:`m_{1}` is the mass of the primary star.

The period is dependent on primary mass while the mass ratio and eccentricity are dependent on both period and primary mass.
Alongside the empirical distributions Moe and Di Stefano gave formulae for the corresponding probability density functions, which Dyad uses to implement the random variables
:class:`dyad.stats.log_period.moe2017`,
:class:`dyad.stats.period.moe2017`,
:class:`dyad.stats.mass_ratio.moe2017`, and 
:class:`dyad.stats.eccentricity.moe2017`.
Since these are dependent on other random variables we must use their shape parameters to fully specify them.
The unit of period is :math:`\mathrm{d}` and the unit of mass is :math:`\mathrm{M}_{\odot}`.

The probability density functions
=================================

Let us plot the PDFs of each random variable in turn.
In each case we will consider the average primary mass for each spectral type, namely :math:`m_{1}/\mathrm{M}_{\odot} = 1, 3.5, 7, 12, 28`, which we can specify now.

.. doctest:: python

   >>> import numpy as np
   >>> m_1 = np.array([1., 3.5, 7., 12, 28.])
   
Log-period
----------

Let us evaluate the conditional PDF of log-period given primary mass, :math:`f_{\log_{10}(P)|M_{1}}`, which is nonzero between :math:`\log_{10}(P/\mathrm{d}) = 0.2` and :math:`\log_{10}(P/\mathrm{d}) = 8`
where :math:`M_{1}/\mathrm{M}_{\odot} \in [0.8, 40]`.
Dyad implements the log-period random variable using the class :class:`dyad.stats.log_period.moe2017`, which has shape parameter ``primary_mass``.

To allow for broadcasting we must increase the dimension of ``m_1`` using the notation ``m_1[:,None]`` to create a new axis.

.. doctest:: python

   >>> import dyad.stats as stats
   >>> log_p = np.linspace(-1., 9.)
   >>> f = stats.log_period.moe2017(m_1[:,None]).pdf(log_p)

Now let us plot these values.
   
.. doctest:: python

   >>> import matplotlib.pyplot as plt
   >>> label = [
   ...     r"$M_{1} = 1\mathrm{M}_{\odot}$",
   ...     r"$M_{1} = 3.5\mathrm{M}_{\odot}$",
   ...     r"$M_{1} = 7\mathrm{M}_{\odot}$",
   ...     r"$M_{1} = 12\mathrm{M}_{\odot}$",
   ...     r"$M_{1} = 28\mathrm{M}_{\odot}$"
   ... ]
   >>> color = ["red", "orange", "green", "blue", "magenta"]
   >>> fig, ax = plt.subplots()
   >>> for (f_i, label_i, color_i) in zip(f, label, color):
   >>>     ax.plot(log_p, f_i, label=label_i, color=color_i)
   >>> ax.legend(frameon=False)
   >>> ax.set_xlabel(r"$\log_{10}(p/\mathrm{d})$")
   >>> ax.set_ylabel(r"$f_{log_{10}(P)}$")
   >>> plt.show()

.. _logperiod:
.. figure:: ./Figures/moe2017_pdf_logperiod.jpg
   :figwidth: 75%
   :align: center

   The conditional PDF of log-period, :math:`\log_{10}(P)` given primary mass, :math:`M_{1}`.

We may also evaluate the conditional PDF of period (rather than log-period)  given primary mass, :math:`f_{P|M_{1}}`, which is nonzero between :math:`P/\mathrm{d} = 10^{0.2}` and :math:`P/\mathrm{d} = 10^{8}`
where :math:`M_{1}/\mathrm{M}_{\odot} \in [0.8, 40]`.

.. doctest:: python
	     
   >>> p = np.logspace(-1., 9.)
   >>> f = stats.period.moe2017(m_1[:,None]).pdf(p)

Mass ratio
----------

Let us evaluate the conditional PDF of mass ratio given log-period and primary mass, :math:`f_{Q|\log_{10}(P), M_{1}}`, which is nonzero between :math:`q = 0.1` and :math:`q = 1`, where :math:`\log_{10}(P/\mathrm{d}) \in [0.2, 8]` and :math:`M_{1}/\mathrm{M}_{\odot} \in [0.8, 40]`.
Dyad implements the mass-ratio random variable using the class :class:`dyad.stats.mass_ratio.moe2017`, which has shape parameters ``log10_period`` and ``primary_mass``.

This PDF is qualitatively different for short- and long-period binary systems.
For short-period systems it is approximately constant except for an excess of twins, this excess being greater for systems with low-mass primary stars than it is for systems with high-mass primary stars.
For long-period systems with high-mass primary stars it is a decreasing function of :math:`q`.
For long-period systems with low-mass primary stars it has a mode at :math:`q = 0.3`.

Let us consider the minimum and maximum allowed log-periods, :math:`0.2` and :math:`8`.
First, the case of :math:`\log_{10}(P/\mathrm{d}) = 0.2`.

.. doctest:: python

   >>> q = np.linspace(0., 1., 500)
   >>> f = stats.mass_ratio.moe2017(0.2, m_1[:,None]).pdf(q)

Which we may plot.

.. doctest:: python

   >>> fig, ax = plt.subplots()
   >>> for (f_i, label_i, color_i) in zip(f, label, color):
   ...     ax.plot(q, f_i, label=label_i, color=color_i)
   >>> ax.legend(frameon=False)
   >>> ax.set_xlabel(r"$q$")
   >>> ax.set_ylabel(r"$f_Q$")
   >>> plt.show()
   
.. _mass_ratio_short_period:
.. figure:: ./Figures/moe2017_pdf_mass_ratio_short_period.jpg
   :figwidth: 75%
   :align: center

   The PDF of mass ratio, :math:`Q`, given log-period, :math:`\log_{10}(P/\mathrm{d}) = 0.2` and primary mass :math:`M_{1}`.

Second, the case of :math:`\log_{10}(P/\mathrm{d}) = 8`.
   
.. doctest:: python

   >>> f = stats.mass_ratio.moe2017(8., m_1[:,None]).pdf(q)

Which we may again plot.
   
.. doctest:: python

   >>> fig, ax = plt.subplots()
   >>> for (f_i, label_i, color_i) in zip(f, label, color):
   ...     ax.plot(q, f_i, label=label_i, color=color_i)
   >>> ax.legend(frameon=False)
   >>> ax.set_xlabel(r"$q$")
   >>> ax.set_ylabel(r"$f_Q$")
   >>> plt.show()
   
.. _mass_ratio_long_period:
.. figure:: ./Figures/moe2017_pdf_mass_ratio_long_period.jpg
   :figwidth: 75%
   :align: center

   The conditional PDF of mass ratio, :math:`Q`, given log-period, :math:`\log_{10}(P/\mathrm{d}) = 8` and primary mass :math:`M_{1}`.
 
Eccentricity
------------

Let us evaluate the conditional PDF of mass ratio given log-period and primary mass, :math:`f_{E|\log_{10}(P), M_{1}}`, which is nonzero between :math:`e = 0.` and :math:`e_{\max}`, where :math:`e_{\max}` is an increasing function of log-period and where :math:`\log_{10}(P/\mathrm{d}) \in [0.9375, 8]` and :math:`M_{1}/\mathrm{M}_{\odot} \in [0.8, 40]`.
Dyad implements the eccentricity random variable using the class :class:`dyad.stats.eccentricity.moe2017`, which has shape parameters ``log10_period`` and ``primary_mass``.
The minimum period is known as the \'circularization period\',
If a binary system has a period shorter than the circularization period then its eccentricity is zero.
Dyad uses a circularation period of :math:`0.9375`, which differs from the value of :math:`0.5` used by Moe and Di Stefano in order to ensure that the PDF always has finite integral.
For a full discussion, see the API documentation.

The PDF is qualitatively different for short- and long-period binary systems.
For short-period systems with small primary star masses it is a descreasing function of :math:`q`.
For short-period systems with large primary star masses it is an increasing function of :math:`q`.
For long-period systems it is also an increasing function of :math:`q`.

Let us consider the log-periods, :math:`\log_{10}(P/\mathrm{d}) = 1` and :math:`\log_{10}(P/\mathrm{d}) = 8`.
First, evaluate the PDF for :math:`\log_{10}(P/\mathrm{d}) = 1`.

.. doctest:: python

   >>> e = np.linspace(0., 1., 500)
   >>> f = stats.eccentricity.moe2017(1., m_1[:,None]).pdf(e)

And plot it.

.. _eccentricity_short_period:
.. figure:: ./Figures/moe2017_pdf_eccentricity_short_period.jpg
   :figwidth: 75%
   :align: center

   The PDF of eccentricity, :math:`E`, given log-period, :math:`\log_{10}(P/\mathrm{d}) = 1` and primary mass, :math:`M_{1}`.

.. doctest:: python

   >>> fig, ax = plt.subplots()
   >>> for (f_i, label_i, color_i) in zip(f, label, color):
   >>>     ax.plot(e, f_i, label=label_i, color=color_i)
   >>> ax.legend(frameon=False)
   >>> ax.set_ylim(0., 5.)
   >>> ax.set_xlabel(r"$e$")
   >>> ax.set_ylabel(r"$f_E$")
   >>> plt.show()

Second, evaluate the PDF for :math:`\log_{10}(P/\mathrm{d}) = 8`.

.. doctest:: python

   >>> f = stats.eccentricity.moe2017(8., m_1[:,None]).pdf(e)

And again plot it.

.. doctest:: python

   >>> fig, ax = plt.subplots()
   >>> for (f_i, label_i, color_i) in zip(f, label, color):
   >>>     ax.plot(e, f_i, label=label_i, color=color_i)
   >>> ax.legend(frameon=False)
   >>> ax.set_xlabel(r"$e$")
   >>> ax.set_ylabel(r"$f_E$")
   >>> plt.show()

.. _eccentricity_long_period:
.. figure:: ./Figures/moe2017_pdf_eccentricity_long_period.jpg
   :figwidth: 75%
   :align: center

   The PDF of eccentricity, :math:`E`, given log-period, :math:`\log_{10}(P/\mathrm{d}) = 8` and primary mass, :math:`M_{1}`.

A complete population
=====================

Let us now synthesize a population of binary systems.
We will use the primary-constrained pairing method to synthesize the primary- and secondary-star masses [K09]_.
According to this method we synthesize the primary-star star mass assuming that it is distributed according to the initial mass function and then synthesize the secondary-star mass using the conditional distribution of secondary-star mass given primary-star mass.

First, speciy a sample size.

.. doctest:: python

   >>> n_binary = 10_000

And sample the primary mass using a Salpeter random variable on the interval :math:`0.8, 40`. 

.. doctest:: python

   >>> m_1 = stats.mass.salpeter1955(0.8, 40.).rvs(size=n_binary)

Next sample the period.

.. doctest:: python

   >>> log_p = stats.log_period.moe2017(m_1).rvs()
   >>> p = 10.**log_p

And the mass ratio. (This can be time consuming. A sample of size :math:`100\,000` may take several minutes or more to be generated.)

.. doctest:: python

   >>> q = stats.mass_ratio.moe2017(log_p, m_1).rvs()

Now sample the eccentricity, remembering that all systems with periods shorter than the circularization period, when :math:`\log_{10}(P) = 0.9375`, have eccentricities of zero.

.. doctest:: python

   >>> e = np.zeros(n_binary)
   >>> idx = (log_p > 0.9375)
   >>> e[idx] = stats.eccentricity.moe2017(log_p[idx], m_1[idx]).rvs()

Finally, sample the longitude of the ascending node, inclination, and argument of periapsis.

.. doctest:: python

   >>> Omega = stats.longitude_of_ascending_node.rvs(size=n_binary)
   >>> i = stats.inclination.rvs(size=n_binary)
   >>> omega = stats.argument_of_pericentre.rvs(size=n_binary)

As before, the class :class:`dyad.TwoBody` can serve as a container for these values.
First convert the periods to their equivalent primary-star semimajor axes.

.. doctest:: python

   >>> import dyad
   >>> a = dyad.semimajor_axis_from_period(p, m_1, m_1*q)
   >>> a_1 = dyad.primary_semimajor_axis_from_semimajor_axis(a, q)

And instantiate a :class:`dyad.TwoBody` object.

.. doctest:: python
	     
   >>> binary = dyad.TwoBody(m_1, q, a_1, e, Omega, i, omega)

Before, we inspected the state of a single member of this population for a given true anomaly.
This time, let us compute the speeds and radii of all primary and secondary stars and plot their histograms.
First, sample the true anomaly.

.. doctest:: python

   >>> theta = stats.true_anomaly(e).rvs()

And compute the primary and secondary radii at these true anomalies.

.. doctest:: python

   >>> r_1 = binary.primary.radius(theta)
   >>> r_2 = binary.secondary.radius(theta)

Now generate the histograms.

.. doctest:: python

   >>> bins = np.logspace(-3., 6., 25)
   >>> edge_r1, count_r1 = np.histogram(r_1, bins=bins)
   >>> edge_r2, count_r2 = np.histogram(r_2, bins=bins)

And finally plot them.

.. doctest:: python

   >>> fig, ax = plt.subplots()
   >>> ax.stairs(edge_r1, count_r1, label="primary")
   >>> ax.stairs(edge_r2, count_r2, label="primary")
   >>> ax.legend(frameon=False)
   >>> ax.set_xscale("log")
   >>> ax.set_xlabel(r"$r/\mathrm{AU}$")
   >>> ax.set_ylabel(r"$\nu$")
   >>> plt.show()

.. _radii:
.. figure:: ./Figures/moe2017_sample_radius.jpg
   :figwidth: 75%
   :align: center

   The histograms of primary and secondary star radii.

Now compute the primary and secondary speeds.
   
.. doctest:: python

   >>> v_1 = binary.primary.speed(theta)
   >>> v_2 = binary.secondary.speed(theta)

Again generate the histograms.

.. doctest:: python

   >>> bins = np.logspace(-3., 6., 25)
   >>> edge_v1, count_v1 = np.histogram(r_1, bins=bins)
   >>> edge_v2, count_v2 = np.histogram(r_2, bins=bins)
   
And again plot them.

.. doctest:: python

   >>> fig, ax = plt.subplots()
   >>> ax.stairs(edge_v1, count_v1, label="primary")
   >>> ax.stairs(edge_v2, count_v2, label="primary")
   >>> ax.legend(frameon=False)
   >>> ax.set_xscale("log")
   >>> ax.set_xlabel(r"$v/\mathrm{km}~\mathrm{s}^{-1}$")
   >>> ax.set_ylabel(r"$\nu$")
   >>> plt.show()

.. _speed:
.. figure:: ./Figures/moe2017_sample_speed.jpg
   :figwidth: 75%
   :align: center

   The histograms of primary and secondary star speeds.
   
References
==========

.. [MS17]

   Moe, Maxwell, and Rosanne Di Stefano. 2017. \'Mind your Ps and Qs:
   the interrelation between period (P) and mass-ratio (Q)
   distributions of binary stars.\' *The Astrophysical Journal
   Supplement Series* 230 (2): 15.

.. [K09]

   Kouwenhoven, M. B. N. et al. 2009. \'Exploring the consequences of pairing
   algorithms for binary stars.\'. *Astronomy & Astrophysics* 493 (3):
   979.
