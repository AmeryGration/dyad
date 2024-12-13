"""
Sane parameters for dyad.stats distributions.

"""

import numpy as np

distcont = [
    ["true_anomaly", (0.5,)],
    ["eccentricity.duquennoy1991", (365.,)],
    ["eccentricity.moe2017", (np.log10(365.), 1.)],
    ["log_period.moe2017", (1.,)],
    ["mass.splitpowerlaw", (0.5, 0.1, 60., -1.3, -2.3)]
    ["mass_ratio.moe2017", (np.log10(365.), 1.)],
    ["period.trunclognorm",
     (np.log(10.)*2.3,
      10.**-2./np.exp(np.log(10.)*4.8),
      10.**12./np.exp(np.log(10.)*4.8),
      np.exp(np.log(10.)*4.8),)],
    ["period.moe2017", (1.,)]
    ["semimajor_axis.opik1924", (15., 10.)],
]
