.. _user_guide:

**********************
Binary-star statistics
**********************

Checking the validity of shape parameters
=========================================

Scipy's random variables do not have a direct method for checking the validity of their shape parameters. This is done behind the scenes (using the private method ``scipy.stats.rv_continuous._argcheck``) only when an attribute is called. When passing an array-like object of shape parameters one invalid element will cause the ``rvs`` attribute to return ``ValueError``. However, in the case that a parameter is invalid the support of the random variable will always be ``[nan, nan]``. We can exploit this fact to check the validity of a shape parameter. 

   >>> mask = np.isfinite(stats.eccentricity.duquennoy1991(p).support()[0])
   >>> e[mask] = stats.eccentricity.duquennoy1991(mask).rvs(size=n)
