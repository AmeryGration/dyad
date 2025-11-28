"""Data to be used when testing Dyad's stats.inclination module

This module is not to be run on its own. It is imported by
`test_stats_inclination.py`.

When checking for value errors each list contains arguments that
should cause such an error. If a value error is raised then the test
is passed.

Each element in a list is itself a list that contains two elements:
1. the arguments for a function being tested, and
2. the desired return value of that function for these arguments.

For example to test the function `dyad.func(*args)` we use the
following data.

.. code-block:: python
   func = [
       [args, val],
       [args, val],
       ...
       [args, val],
   ]

If the function `dyad.stats.inclination.{attribute,method}(*args)` returns
the value `val` then the test is passed.

"""
inclination_pdf = [
    [[0.0], 0.0],
    [[3.141592653589793], 6.123233995736766e-17],
    [[0.22862176032196072], 0.11331768088728117],
    [[[0.0, 1.57079633, 3.14159265, 4.71238898, 6.28318531]], [0.0, 0.5, 1.7948965149208059e-09, 0.0, 0.0]],
]
inclination_cdf = [
    [[0.0], 0.0],
    [[3.141592653589793], 1.0],
    [[0.22862176032196072], 0.013010161093346517],
    [[[0.0, 1.57079633, 3.14159265, 4.71238898, 6.28318531]], [0.0, 0.5000000016025518, 1.0, 1.0, 1.0]],
]
inclination_ppf = [
    [[0.0], 0.0],
    [[1.0], 3.141592653589793],
    [[0.013010161093346517], 0.22862176032196074],
    [[[0.0, 0.5000000016025518, 1.0, 1.0, 1.0]], [0.0, 1.57079633, 3.141592653589793, 3.141592653589793, 3.141592653589793]],
]
inclination_rvs = [
    [[0.0, 1.0, 1, 93438701], [1.110319376443364]],
    [[0.0, 1.0, 2, 93438701], [1.110319376443364, 1.6605765617347878]],
    [[0.0, 1.0, (2, 3), 93438701], [[1.110319376443364, 1.6605765617347878, 1.6689160101642333], [0.7570532617022121, 2.087436433454325, 1.9339911408202142]]],
]
