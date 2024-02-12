__all__ = [
    "trunclognorm"
]

import numpy as np
import scipy as sp

def _lazywhere(cond, arrays, f, fillvalue=None, f2=None):
    cond = np.asarray(cond)
    if fillvalue is None:
        if f2 is None:
            raise ValueError("One of (fillvalue, f2) must be given.")
        else:
            fillvalue = np.nan
    else:
        if f2 is not None:
            raise ValueError("Only one of (fillvalue, f2) can be given.")

    args = np.broadcast_arrays(cond, *arrays)
    cond, arrays = args[0], args[1:]
    temp = tuple(np.extract(cond, arr) for arr in arrays)
    tcode = np.mintypecode([a.dtype.char for a in arrays])
    out = np.full(np.shape(arrays[0]), fill_value=fillvalue, dtype=tcode)
    np.place(out, cond, f(*temp))
    if f2 is not None:
        temp = tuple(np.extract(~cond, arr) for arr in arrays)
        np.place(out, ~cond, f2(*temp))

    return out

def _lognorm_logpdf(x, s):
    # Replica of ~scipy.stats._continuous_distns.cd._lognorm_logpdf~,
    # which is called by ~scipy.stats.lognorm~.
    return _lazywhere(
        x != 0,
        (x, s),
        lambda x, s: -np.log(x)**2./(2.*s**2.) - np.log(s*x*np.sqrt(2*np.pi)),
        -np.inf
    )


class _trunclognorm_gen(sp.stats.rv_continuous):
    r"""The truncated lognormal random variable

    %(before_notes)s

    See Also
    --------
    scipy.stats.lognorm, scipy.stats.truncnorm

    Notes
    -----
    The probability density function for `trunclognorm` is:

    .. math::

        f(x, a, b, s) = \dfrac{\exp\left(-\dfrac{\log^{2}(x)}{2s^{2}}\right)}{\erf\left(\dfrac{\log(b)}{s}\right) - \erf\left(\dfrac{\log(a)}{s}\right)}

    where :math:`a < x < 0`, :math:`0 < a < b`, and :math:`s > 0` [1]_.

    `trunclognorm` takes :math:`a`, :math:`b`, and :math:`c` as shape
    parameters.

    Notice that the truncation values, :math:`a` and :math:`b`, are
    defined in standardized form:

    .. math::
    
        a = (u_{\mathrm{l}} - \mathrm{loc})/\mathrm{scale}, 
        b = (u_{\mathrm{r}} - \mathrm{loc})/\mathrm{scale}

    where :math:`u_{\mathrm{l}}` and :math:`u_{mathrm{r}` are the
    specific left and right truncation values, respectively. In other
    words, the support of the distribution becomes
    :math:`(a*\mathrm{scale} + \mathrm{loc}) < x <= (b*\mathrm{scale}
    + \mathrm{loc})` when :math:`\mathrm{loc}` and/or
    :math:`\mathrm{scale}` are provided.
    
    %(after_notes)s
    
    Suppose a normally distributed random variable ``X`` has  mean ``mu`` and
    standard deviation ``sigma``. Then ``Y = exp(X)`` is lognormally
    distributed with ``s = sigma`` and ``scale = exp(mu)``.
    
    References
    ----------
    .. [1] Reference

    %(example)s
    
    """ 
    def _argcheck(self, s, a, b):
        return (a >= 0.) & (b > a) & (s > 0.)

    def _shape_info(self):
        is_ = _ShapeInfo("s", False, (0, np.inf), (False, False))
        ia = _ShapeInfo("a", False, (0, np.inf), (True, False))
        ib = _ShapeInfo("b", False, (0, np.inf), (False, False))
        return [is_, ia, ib]

    # def _fitstart(self, data):
    #     pass
    #     # # Arbitrary, but default a=b=c=1 is not valid
    #     # return super()._fitstart(data, args=(1, 0, 1)) # ???
    
    def _get_support(self, s, a, b):
        return a, b

    def _pdf(self, x, s, a, b):
        A = sp.special.ndtr(np.log(b)/s) - sp.special.ndtr(np.log(a)/s)

        return np.exp(_lognorm_logpdf(x, s))/A

    def _cdf(self, x, s, a, b):
        A = sp.special.ndtr(np.log(b)/s) - sp.special.ndtr(np.log(a)/s)
        
        return (sp.special.ndtr(np.log(x)/s) - sp.special.ndtr(np.log(a)/s))/A

    def _ppf(self, q, s, a, b):
        A = sp.special.ndtr(np.log(b)/s) - sp.special.ndtr(np.log(a)/s)

        return np.exp(
            s*sp.stats.norm.ppf(A*q + sp.special.ndtr(np.log(a)/s))
        )


trunclognorm = _trunclognorm_gen(name="trunclognorm")
