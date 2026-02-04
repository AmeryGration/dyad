# Package data

The probability density function and cumulative distribution
function for the log-period random variable defined by
Moe and Stefano (2017) do not have closed-form expression (see
dyad.stats.period.moe2017 and dyad.stats.log_period.moe2017). Dyad
evaluates these functions by interpolating between values pre-computed
on a regular lattice of arguments.

This directory contains all these pre-computed values and the code
used to generate them.
