import numpy as np
from scipy.stats import rv_continuous


class kroupa2001_gen(rv_continuous):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._interpolator = None

    def interpolator(self, a, b):
        def fun(x):
            return 1./(b - a)

        print(a, b)
        if self._pdf_interp is None:
            self._pdf_interp = fun

        return self._interpolator

    def _get_support(self, a, b):
        res = (a, b)

        return res

    def _pdf(self, x, a, b):
        res = self.interpolator(a, b)(x)

        return res


kroupa2001 = kroupa2001_gen(name="kroupa2001")

rv_mass = kroupa2001(a=0.1, b=1.1)
print(rv_mass.pdf(1.))

rv_mass = kroupa2001(a=0.1, b=3.1)
print(rv_mass.pdf(1.))
