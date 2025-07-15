#!/usr/bin/env python

r"""Sample of the period distribution given by Moe and Stefano (2017)

Neither the probability density function (PDF) nor cumulative
distribution function (CDF) for the period random variables
defined by Moe and Stefano (2017) have closed-form expression. Dyad
evaluates these functions by interpolating between values pre-computed
on a regular lattice of arguments. The CDF is computed by integrating
the PDF using the trapezium rule with nodes placed on the points of
this lattice. Moe and Stefano give the observed frequency of
period rather than its PDF. To compute the latter we must
normalize the former. This script performs the reqired sampling,
integration, and normalization, saving the results to file. Those
files are read by `dyad/stats/period.py`, which implements the
random variable `dyad.stats.period.moe2017` by performing the
interpolation.

The period may be computed using the log-period according to the
formula given in the docstring for `dyad.stats.period.moe2017`. This
script therefore reads the data already generated for use by
`dyad.stats.log_period.moe2017`.

"""
import numpy as np

from importlib.resources import files
from scipy.integrate import cumulative_trapezoid
from dyad.stats import mass_ratio

#############################################################################
# Create grid of sample points
#############################################################################
path = "dyad.stats.data.log_period"
with files(path).joinpath("log10_period_sample.dat") as f_name:
    log10_period_sample = np.loadtxt(f_name)
with files(path).joinpath("primary_mass_sample.dat") as f_name:
    primary_mass_sample = np.loadtxt(f_name)
with files(path).joinpath("frequency_sample.dat") as f_name:
    frequency_sample = np.loadtxt(f_name)
with files(path).joinpath("cumulative_frequency_sample.dat") as f_name:
    cumulative_frequency_sample = np.loadtxt(f_name)

period_sample = 10.**log10_period_sample

#############################################################################
# Sample the frequency function using a rectilinear lattice 
#############################################################################
frequency_sample = frequency_sample/period_sample

#############################################################################
# Sample the cumulative frequency function using a rectilinear lattice 
#############################################################################
cumulative_frequency_sample = cumulative_trapezoid(
    frequency_sample, period_sample, initial=0.
)

#############################################################################
# Compute equivalent sample of the PDF and CDF
#############################################################################
frequency_sample = frequency_sample/cumulative_frequency_sample[:,-1:]
cumulative_frequency_sample = (
    cumulative_frequency_sample/cumulative_frequency_sample[:,-1:]
)

#############################################################################
# Save data
#############################################################################
np.savetxt(
    "./primary_mass_sample.dat", primary_mass_sample
)
np.savetxt(
    "./period_sample.dat", period_sample
)
np.savetxt(
    "./frequency_sample.dat", frequency_sample
)
np.savetxt(
    "./cumulative_frequency_sample.dat", cumulative_frequency_sample
)

# _moe2017_pdf_interp = RegularGridInterpolator(
#     (period_sample, primary_mass_sample),
#     frequency_sample.T
# )
# _moe2017_cdf_interp = RegularGridInterpolator(
#     (period_sample, primary_mass_sample),
#     cumulative_frequency_sample.T
# )
