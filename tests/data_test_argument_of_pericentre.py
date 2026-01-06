"""Data to be used when testing Dyad's stats.argument_of_pericentre module

This module is not to be run on its own. It is imported by
`test_stats_argument_of_pericentre.py`.

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

If the function `dyad.stats.argument_of_pericentre.{attribute,method}(*args)` returns
the value `val` then the test is passed.

"""
argument_of_pericentre_pdf = [
    [[0.0], 0.15915494309189535],
    [[6.283185307179586], 0.15915494309189535],
    [[1.634480683188948], 0.15915494309189535],
    [[[0.0, 1.5707963267948966, 3.141592653589793, 4.71238898038469, 6.283185307179586]], [0.15915494309189535, 0.15915494309189535, 0.15915494309189535, 0.15915494309189535, 0.15915494309189535]],
]
argument_of_pericentre_cdf = [
    [[0.0], 0.0],
    [[6.283185307179586], 1.0],
    [[1.634480683188948], 0.26013568011773924],
    [[[0.0, 1.5707963267948966, 3.141592653589793, 4.71238898038469, 6.283185307179586]], [0.0, 0.25, 0.5, 0.75, 1.0]],
]
argument_of_pericentre_ppf = [
    [[0.0], 0.0],
    [[1.0], 6.283185307179586],
    [[0.26013568011773924], 1.634480683188948],
    [[[0.0, 0.25, 0.5, 0.75, 1.0]], [0.0, 1.5707963267948966, 3.141592653589793, 4.71238898038469, 6.283185307179586]],
]
argument_of_pericentre_rvs = [
    [[0.0, 1.0, 1, 93438701], [1.745546070382502]],
    [[0.0, 1.0, 2, 93438701], [1.745546070382502, 3.4232668186370896]],
    [[0.0, 1.0, (2, 3), 93438701], [[1.745546070382502, 3.4232668186370896, 3.449350353355953], [0.8580854061275839, 4.693418657993533, 4.257682546194693]]],
]
