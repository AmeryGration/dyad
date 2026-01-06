"""Data to be used when testing Dyad's stats.longitude_of_ascending_node module

This module is not to be run on its own. It is imported by
`test_stats_longitude_of_ascending_node.py`.

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

If the function `dyad.stats.longitude_of_ascending_node.{attribute,method}(*args)` returns
the value `val` then the test is passed.

"""
longitude_of_ascending_node_pdf = [
    [[0.0], 0.15915494309189535],
    [[6.283185307179586], 0.15915494309189535],
    [[0.45724352064392143], 0.15915494309189535],
    [[[0.0, 1.57079633, 3.14159265, 4.71238898, 6.28318531]], [0.15915494309189535, 0.15915494309189535, 0.15915494309189535, 0.15915494309189535, 0.0]],
]
longitude_of_ascending_node_cdf = [
    [[0.0], 0.0],
    [[6.283185307179586], 1.0],
    [[0.45724352064392143], 0.07277256650722119],
    [[[0.0, 1.57079633, 3.14159265, 4.71238898, 6.28318531]], [0.0, 0.25000000051010807, 0.49999999942866674, 0.7499999999387748, 1.0]],
]
longitude_of_ascending_node_ppf = [
    [[0.0], 0.0],
    [[1.0], 6.283185307179586],
    [[0.07277256650722119], 0.45724352064392143],
    [[[0.0, 0.25000000051010807, 0.49999999942866674, 0.7499999999387748, 1.0]], [0.0, 1.57079633, 3.14159265, 4.71238898, 6.283185307179586]],
]
longitude_of_ascending_node_rvs = [
    [[0.0, 1.0, 1, 93438701], [1.745546070382502]],
    [[0.0, 1.0, 2, 93438701], [1.745546070382502, 3.4232668186370896]],
    [[0.0, 1.0, (2, 3), 93438701], [[1.745546070382502, 3.4232668186370896, 3.449350353355953], [0.8580854061275839, 4.693418657993533, 4.257682546194693]]],
]
