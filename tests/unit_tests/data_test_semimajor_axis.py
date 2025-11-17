"""Data to be used when testing Dyad's stats.semimajor_axis module

This module is not to be run on its own. It is imported by
`test_stats_semimajor_axis.py`.

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

If the function `dyad.stats.semimajor_axis.{attribute,method}(*args)` returns
the value `val` then the test is passed.

"""
opik1924_pdf = [
    [[0.0, 0.1, 1.0], 0.0],
    [[1.0, 0.1, 1.0], 0.43429448190325187],
    [[0.21531557269370072, 0.1, 1.0], 2.017013801974564],
    [[[0.1, 0.25, 0.5, 0.75, 1.0], 0.1, 1.0], [4.3429448190325175, 1.7371779276130073, 0.8685889638065037, 0.5790593092043358, 0.43429448190325187]],
]
opik1924_cdf = [
    [[0.0, 0.1, 1.0], 0.0],
    [[1.0, 0.1, 1.0], 1.0],
    [[0.21531557269370072, 0.1, 1.0], 0.3330754412975461],
    [[[0.1, 0.25, 0.5, 0.75, 1.0], 0.1, 1.0], [0.0, 0.39794000867203755, 0.6989700043360187, 0.8750612633917001, 1.0]],
]
opik1924_ppf = [
    [[0.0, 0.1, 1.0], 0.1],
    [[1.0, 0.1, 1.0], 1.0],
    [[0.3330754412975461, 0.1, 1.0], 0.21531557269370072],
    [[[0.0, 0.39794000867203755, 0.6989700043360187, 0.8750612633917001, 1.0], 0.1, 1.0], [0.1, 0.25, 0.49999999999999994, 0.7500000000000001, 1.0]],
]
opik1924_rvs = [
    [[0.1, 1.0, 0.0, 1.0, 1, 93438701], [0.1895886287778149]],
    [[0.1, 1.0, 0.0, 1.0, 2, 93438701], [0.1895886287778149, 0.3506144700591096]],
    [[0.1, 1.0, 0.0, 1.0, (2, 3), 93438701], [[0.1895886287778149, 0.3506144700591096, 0.35398198410467774], [0.13695204889605295, 0.5584454790558564, 0.476026602085919]]],
]
