"""A binary kinematics module.

"""

__all__ = [
    "Body",
]

import numpy as np
import scipy as sp

from astropy import constants

GRAV_CONST = 6.6743e-08 # Units: cgs
M_SUN = 1.98840987e+33 # Units: cgs
R_SUN = 6.9570000000e10 # Units: cgs
L_SUN = 3.828e+33 # Units: cgs
SIGMA = 5.6703744191844314e-05 # Units: cgs
KSB = 4.*np.pi*SIGMA # Units: cgs

# GRAV_CONST = constants.G.cgs.to_value() # 6.6743e-08 # G (cgs)
# M_SUN = constants.M_sun.cgs.to_value() # 1.989e33 # Msun (cgs)
# R_SUN = constants.R_sun.cgs.to_value() # 6.9566e+10 # Rsun (cgs)
# L_SUN = constants.L_sun.cgs.to_value() # 3.8515e+33 # Lsun (cgs)
# SIGMA = constants.sigma_sb.cgs.to_value()  # 5.670352798e-5 # Stefan-Boltzmann constant (cgs)
# KSB = 4.*np.pi*SIGMA # 4pi * Stefan-Boltzmann - used often

def hrd(m):
    """
    Function to evaluate zero-age main sequence location of
    a hydrogen-burning star of mass m, given Z=0.02.
     (from Tout et al., 1996, MNRAS, 281, 257, binary_c version)

    Input :
    * m = Mass (in solar unit)

    Output :
    (Teff, L, g) tuples (units: Kelvin, Lsun, gsun)
    """
    ms_parameter = (
        0.,
        0.397042,
        8.52763,
        0.00025546,
        5.43289,
        5.56358,
        0.788661,
        0.00586685,
        1.71536,
        6.59779,
        10.0885,
        1.01249,
        0.0749017,
        0.0107742,
        3.08223,
        17.8478,
        0.00022582
    )

    num = ms_parameter[1]*m**5.*m**0.5 + ms_parameter[2]*m**8.*m**3.
    denom = (
        ms_parameter[3] + m**3.
        + ms_parameter[4]*m**5.
        + ms_parameter[5]*m**5.*m**2.
        + ms_parameter[6]*m**8.
        + ms_parameter[7]*m**8.*m*m**0.5
    )
    L = num/denom

    num = (
        ms_parameter[8]*m**2.5
        + ms_parameter[9]*m**6.5
        + ms_parameter[10]*m**2.*m**9.
        + ms_parameter[11]*m**19.
        + ms_parameter[12]*m**19.5
    )
    denom = (
        ms_parameter[13]
        + m**2.*(ms_parameter[14] + m**6.5*(ms_parameter[15] + m**9.*m))
        + ms_parameter[16]*m**19.5
    )
    R = num/denom

    g = GRAV_CONST*(m*M_SUN)/(R*R_SUN)**2.

    T_eff = ((L*L_SUN)/(KSB*(R*R_SUN)**2))**0.25

    return (T_eff, L, g, R)


class Body:
    def __init__(self, m):
        """Properties of zero-age main-sequence stars"""
        self._mass = m
        self._hrd = hrd(m)
        # Properties to include:
        # spectral type,
        # luminosity,
        # absolute_magnitude,
        # radius,
        # metallicity,
        # age,
        # colors,

    # To compute color use, for example, `https://pyastronomy.readthedocs.io/en/latest/pyaslDoc/aslDoc/aslExt_1Doc/ramirez2005.html` or `https://github.com/casaluca/colte`

    @property
    def mass(self):
        return self._mass

    @property
    def T_eff(self):
        return self._hrd[0]

    @property
    def luminosity(self):
        return self._hrd[1]

    @property
    def surface_gravity(self):
        return self._hrd[2]

    @property
    def radius(self):
        return self._hrd[3]
