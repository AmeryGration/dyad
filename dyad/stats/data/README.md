# Package data

The probability density functions and cumulative distribution
functions for the period and log-period random variables defined by
Moe and Stefano (2017) do not have closed-form expression (see
dyad.stats.period.moe2017 and dyad.stats.log_period.moe2017). Dyad
evaluates these functions by interpolating between values pre-computed
on a regular lattice of arguments.

The probability density function and cumulative distribution function
for secondary mass implied by the distributions for mass ratio and
period given by Moe and Stefano (2017) must also be evaluated by
interpolating between values pre-computed on a regular lattice of
arguments.

This directory contains all these pre-computed values and the code
used to generate them.
