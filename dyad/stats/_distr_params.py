"""
Sane parameters for dyad.stats distributions.

"""

import numpy as np

distcont = [
    ["true_anomaly", (0.5,)],
    # ["eccentricity.powerlaw", (0.659,)],
    ["eccentricity.duquennoy1991", (365.,)],
    ["eccentricity.moe2017", ((2.56, 1.),)],
    ["log_period.moe2017", (1.,)],
    ["mass.splitpowerlaw", ((0.5, 0.1, 60., -1.3, -2.3),)],
    ["mass_ratio.moe2017", ((2.56, 1.),)],
    ["period.trunclognorm", ((5.30, 1.58e-07, 15800000., 63100.),)],
    ["period.duquennoy1991", (1.,)],
    ["period.moe2017", (1.,)],
    # ["semimajor_axis.opik1924", ((15., 10.),)],
]

distdiscrete = []

# invdistcont = [
#     ["true_anomaly", (1.,)],
#     ["eccentricity.duquennoy1991", ()],
#     ["eccentricity.moe2017", ()],
#     ["log_period.moe2017", ()],
#     ["mass.splitpowerlaw", ()],
#     ["mass_ratio.moe2017", ()],
#     ["period.trunclognorm", ()],
#     ["period.moe2017", ()],
#     ["semimajor_axis.opik1924", ()],
# ]
