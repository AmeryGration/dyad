# replacement for _lazywhere which has been
# deprecated from scipy

import numpy as np

try:
    # scipy <= 1.15
    from scipy._lib._array_api import array_namespace
except ModuleNotFoundError:
    # later scipy
    class _NumpyArrayNamespace:
        bool = np.bool_
        bool_ = np.bool_

        def __getattr__(self, name):
            return getattr(np, name)

        @staticmethod
        def astype(x, dtype, *, copy=True):
            return np.asarray(x).astype(dtype, copy=copy)

        @staticmethod
        def asarray(x, dtype=None, *, copy=None):
            if copy is None:
                return np.asarray(x, dtype=dtype)
            return np.array(x, dtype=dtype, copy=copy)

    _numpy_array_namespace = _NumpyArrayNamespace()

    def array_namespace(*arrays):
        return _numpy_array_namespace




def _lazywhere(cond, arrays, f, fillvalue=None, f2=None):
    """Return elements chosen from two possibilities depending on a condition

    Equivalent to ``f(*arrays) if cond else fillvalue`` performed elementwise.

    Parameters
    ----------
    cond : array
        The condition (expressed as a boolean array).
    arrays : tuple of array
        Arguments to `f` (and `f2`). Must be broadcastable with `cond`.
    f : callable
        Where `cond` is True, output will be ``f(arr1[cond], arr2[cond], ...)``
    fillvalue : object
        If provided, value with which to fill output array where `cond` is
        not True.
    f2 : callable
        If provided, output will be ``f2(arr1[cond], arr2[cond], ...)`` where
        `cond` is not True.

    Returns
    -------
    out : array
        An array with elements from the output of `f` where `cond` is True
        and `fillvalue` (or elements from the output of `f2`) elsewhere. The
        returned array has data type determined by Type Promotion Rules
        with the output of `f` and `fillvalue` (or the output of `f2`).

    Notes
    -----
    ``xp.where(cond, x, fillvalue)`` requires explicitly forming `x` even where
    `cond` is False. This function evaluates ``f(arr1[cond], arr2[cond], ...)``
    onle where `cond` ``is True.

    Examples
    --------
    >>> import numpy as np
    >>> a, b = np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])
    >>> def f(a, b):
    ...     return a*b
    >>> _lazywhere(a > 2, (a, b), f, np.nan)
    array([ nan,  nan,  21.,  32.])

    """
    xp = array_namespace(cond, *arrays)

    if (f2 is fillvalue is None) or (f2 is not None and fillvalue is not None):
        raise ValueError("Exactly one of `fillvalue` or `f2` must be given.")

    args = xp.broadcast_arrays(cond, *arrays)
    bool_dtype = xp.asarray([True]).dtype  # numpy 1.xx doesn't have `bool`
    cond, arrays = xp.astype(args[0], bool_dtype, copy=False), args[1:]

    temp1 = xp.asarray(f(*(arr[cond] for arr in arrays)))

    if f2 is None:
        # If `fillvalue` is a Python scalar and we convert to `xp.asarray`, it gets the
        # default `int` or `float` type of `xp`, so `result_type` could be wrong.
        # `result_type` should/will handle mixed array/Python scalars;
        # remove this special logic when it does.
        if type(fillvalue) in {bool, int, float, complex}:
            with np.errstate(invalid='ignore'):
                dtype = (temp1 * fillvalue).dtype
        else:
           dtype = xp.result_type(temp1.dtype, fillvalue)
        out = xp.full(cond.shape, dtype=dtype,
                      fill_value=xp.asarray(fillvalue, dtype=dtype))
    else:
        ncond = ~cond
        temp2 = xp.asarray(f2(*(arr[ncond] for arr in arrays)))
        dtype = xp.result_type(temp1, temp2)
        out = xp.empty(cond.shape, dtype=dtype)
        out[ncond] = temp2

    out[cond] = temp1

    return out
