"""Data to be used when testing Dyad's stats.secondary_mass module

This module is not to be run on its own. It is imported by
`test_stats_secondary_mass.py`.

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

If the function `dyad.stats.secondary_mass.{attribute,method}(*args)` returns
the value `val` then the test is passed.

"""
uniform_pdf = [
    [[0.08, 10.0, 0.08, 0.1], 0.0],
    [[150.0, 10.0, 0.08, 0.1], 0.0],
    [[1.331805596501795, 10.0, 0.08, 0.1], 0.1111111111111111],
    [[[0.0, 10.0, 25.0, 50.0, 100.0, 150.0], 10.0, 0.08, 0.1], [0.0, 0.1111111111111111, 0.0, 0.0, 0.0, 0.0]],
]
uniform_cdf = [
    [[0.08, 10.0, 0.08, 0.1], 0.0],
    [[150.0, 10.0, 0.08, 0.1], 1.0],
    [[1.331805596501795, 10.0, 0.08, 0.1], 0.03686728850019944],
    [[[0.0, 10.0, 25.0, 50.0, 100.0, 150.0], 10.0, 0.08, 0.1], [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
]
uniform_ppf = [
    [[0.0, 10.0, 0.08, 0.1], 1.0],
    [[1.0, 10.0, 0.08, 0.1], 10.0],
    [[0.03686728850019944, 10.0, 0.08, 0.1], 1.331805596501795],
    [[[0.0, 1.0, 1.0, 1.0, 1.0, 1.0], 10.0, 0.08, 0.1], [1.0, 10.0, 10.0, 10.0, 10.0, 10.0]],
]
uniform_rvs = [
    [[10.0, 0.08, 0.1, 0.0, 1.0, 1, 93438701], [3.500310569464078]],
    [[10.0, 0.08, 0.1, 0.0, 1.0, 2, 93438701], [3.500310569464078, 5.903468521377036]],
    [[10.0, 0.08, 0.1, 0.0, 1.0, (2, 3), 93438701], [[3.500310569464078, 5.903468521377036, 5.940830432731382], [2.2291168058219935, 7.722827014774605, 7.098681010086752]]],
]
