title: 'Dyad: a binary-star dynamics and statistics package for Python'
tags:
  - Python
  - astronomy
  - astrodynamics
  - binary stars
  - population synthesis
authors:
  - name: Amery Gration
    orcid: 0000-0003-1379-6696
    affiliation: 1
affiliations:
  - name: Astrophysics Research Group, University of Surrey, Guildford, GU2 7XH, United Kingdom
	index: 1
    ror: 00ks66431
date: 3 March 2025
bibliography: paper.bib
---

# Summary

Dyad is a pure-Python two-body dynamics and binary-star statistics
package for astrophysicists. It allows the user to compute the
kinematic properties of a bound gravitational two-body system given
that system's component masses and orbital elements. It also allows
the user to synthesize a population of binary stars with component
masses and orbital elements that follow a given
distribution. Specifically, Dyad allows the user to synthesize
primary- and secondary-star masses in a manner consistent with given
distributions of stellar masses and mass ratios. It does so by
implementing the method of @gration2025b. Accordingly, Dyad provides a
library of distributions for stellar mass, mass-ratio, and the orbital
elements. This library includes (but is not limited to) the
distributions for (1) the initial stellar mass published by
@chabrier2003, @kroupa2001, and @salpeter1955, and (2) the
mass-ratios and orbital elements of binary stars in the Solar
neighbourhood published by @duquennoy1991 and @moe2017.

# Statement of need

I wrote Dyad to implement the work on binary-star population dynamics
and stellar population synthesis presented by @gration2025a and
@gration2025b. Although binary-star population dynamics is an active
area of research [see, for example, @minor2010; @rastello2020; and
@arroyo-polonio2023] there is no publicly available software to
implement it. To compute the kinematic properties of binary systems
researchers new to the subject must write their own software. I hope
that Dyad fills this gap. Stellar population synthesis (i.e. the
modelling of the evolution of populations of stellar systems) is also
an active area of research [@izzard2019]. However, the software is
better developed. In order to run population synthesis programmes the
user must provide a description of the initial unevolved
population. This can be done by sampling the appropriate random
variables. A number of packages allow the user to do this. Some [such
as *Binary_c-python* by @hendriks2023; or *IMF* by @ginsburg2021]
provide a probability density function, which can be used by an
out-of-package sampling routine (such as rejection sampling or
Markov-chain Monte-Carlo sampling). Others [such as COSMIC by
@breivik2020; or COMPAS by @riley2022] do not provide the probability
density function explicitly but allow the user to generate samples
directly. Uniquely, Dyad implements these distributions as instances
of the *Scipy* random variable class `scipy.stats.rv_continous`,
allowing the full functionality provided by that class. This includes
the evaluation of the probability density function and the generation
of samples of a given size.

# References
