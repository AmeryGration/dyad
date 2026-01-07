"""
Sane parameters for dyad.stats distributions.

"""

import numpy as np

distcont = [
    ["true_anomaly", (0.5,)],
    ["eccentricity.powerlaw", (0.5,)],
    ["eccentricity.duquennoy1991", (365.,)],
    ["eccentricity.moe2017", ((2.56, 1.),)],
    ["log_period.moe2017", (1.,)],
    ["mass.salpeter1955", ((0.4, 10.),)],
    ["mass.kroupa2002", ((0.08, 150.),)],
    ["mass.splitpowerlaw", ((0.5, 0.08, 150., -1.3, -2.3),)],
    ["mass_ratio.moe2017", ((2.56, 1.),)],
    ["mass_ratio.uniform", ((100., 0.08, 0.1),)],
    ["semimajor_axis.opik1924", ((1_000., 10_000.),)],
]

distdiscrete = []
