__all__ = [
    "opik1924",
]

import numpy as np
import scipy as sp

class _opik1924_gen(sp.stats._continuous_distns.reciprocal_gen):
    r"""An Öpik (1924) semimajor axis random variable

    """
    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)

opik1924 = _opik1924_gen(name="opik1924")
