__all__ = [
    "opik1924",
]

import numpy as np
import scipy as sp

class _opik1924_gen(sp.stats._continuous_distns.reciprocal_gen):
    r"""An Öpik-type semimajor axis random variable

    """
    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)

opik1924 = _opik1924_gen(name="opik1924")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Plot loguniform semimajor axis
    semimajor_axis = opik1924(1.e3, 3.e3).rvs(size=10_000)
    counts, edges = np.histogram(
        np.log10(semimajor_axis),
        bins=25,
        density=True
    )
    print(np.min(semimajor_axis), np.max(semimajor_axis))
    
    fig, ax = plt.subplots()
    ax.stairs(counts, edges)
    ax.set_xlabel(r"$\log_{10}(a/\mathrm{AU})$")
    ax.set_ylabel(r"$\hat{f}$")
    plt.show()
    
