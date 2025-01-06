"""Data to be used when testing Dyad

This module is not to be run on its own. It is imported by `unit_test.py`.

When checking for value errors each list contains arguments that should cause such an error. If a value error is raised then the test is passed.

Each element in a list is itself a list that contains two elements:
1. the arguments for a function being tested, and
2. the desired return value of that function for these arguments.

For example to test the function `dyad.func(*args)` we use the following data.

.. code-block:: python
   func = [
       [args, val],
       [args, val],
       ...
       [args, val],
   ]

If the function `dyad.func(*args)` returns the value `val` then the test is passed.

"""

import numpy as np

check_real = ["1", 1.j, 1. + 1.j] # Invalid type
check_nonnegative = [-1, 0.] # Invalid value

check_mass_type = check_real
check_mass_value = check_nonnegative

check_eccentricity_type = check_real
check_eccentricity_value = [-1, 1., 2.] # Invalid eccentricity

check_semimajor_axis_type = check_real
check_semimajor_axis_value = check_nonnegative

check_period_type = check_real
check_period_value = check_nonnegative

semimajor_axis_from_period = [
    # [[p, m_1, m_2], a],
    [[1., 1., 1.], 0.024657235354280858],
]

period_from_semimajor_axis = [
    # [[p, m_1, m_2], a],
    [[0.024657235354280858, 1., 1.], 1.],
]

mean_anomaly_from_eccentric_anomaly = [
    # [[        mu,   e],      theta]
    [[-0.5*np.pi, 0.], -0.5*np.pi],
    [[ 0.       , 0.],  0.       ],
    [[ 0.5*np.pi, 0.],  0.5*np.pi],
    [[     np.pi, 0.],      np.pi],
    [[ 1.5*np.pi, 0.],  1.5*np.pi],
    [[ 2.0*np.pi, 0.],  2.0*np.pi],
    [[ 2.5*np.pi, 0.],  2.5*np.pi],
    [[-0.5*np.pi, 0.5], -1.07079633],
    [[ 0.       , 0.5],  0.        ],
    [[ 0.5*np.pi, 0.5],  1.07079633],
    [[     np.pi, 0.5],     np.pi  ],
    [[ 1.5*np.pi, 0.5],  5.21238898],
    [[ 2.0*np.pi, 0.5],  6.28318531],
    [[ 2.5*np.pi, 0.5],  7.35398163],
]

eccentric_anomaly_from_true_anomaly = [
    # [[     theta,   e],      theta]
    [[-0.5*np.pi, 0.], -0.5*np.pi],
    [[ 0.       , 0.],  0.       ],
    [[ 0.5*np.pi, 0.],  0.5*np.pi],
    [[     np.pi, 0.],      np.pi],
    [[ 1.5*np.pi, 0.],  1.5*np.pi],
    [[ 2.0*np.pi, 0.],  2.0*np.pi],
    [[ 2.5*np.pi, 0.],  2.5*np.pi],
    [[-0.5*np.pi, 0.5], -1.0471975511965983],
    [[ 0.       , 0.5],  0.                ],
    [[ 0.5*np.pi, 0.5],  1.0471975511965983],
    [[     np.pi, 0.5],      np.pi             ],
    [[ 1.5*np.pi, 0.5],  5.235987755982988 ],
    [[ 2.0*np.pi, 0.5],  2.0*np.pi         ],
    [[ 2.5*np.pi, 0.5],  7.3303828583761845],
]

true_anomaly_from_eccentric_anomaly = [
    # [[        mu,   e],      theta]
    [[-0.5*np.pi, 0.], -0.5*np.pi],
    [[ 0.       , 0.],  0.       ],
    [[ 0.5*np.pi, 0.],  0.5*np.pi],
    [[     np.pi, 0.],      np.pi],
    [[ 1.5*np.pi, 0.],  1.5*np.pi],
    [[ 2.0*np.pi, 0.],  2.0*np.pi],
    [[ 2.5*np.pi, 0.],  2.5*np.pi],
    [[-0.5*np.pi, 0.5], -2.0943951023931966],
    [[ 0.       , 0.5],  0.                ],
    [[ 0.5*np.pi, 0.5],  2.0943951023931966],
    [[     np.pi, 0.5],      np.pi         ],
    [[ 1.5*np.pi, 0.5],  4.1887902047863905],
    [[ 2.0*np.pi, 0.5],  2.0*np.pi         ],
    [[ 2.5*np.pi, 0.5],  8.377580409572783 ],
]

mean_anomaly_from_true_anomaly = [
    # [[     theta,   e],      theta]
    [[-0.5*np.pi, 0.], -0.5*np.pi],
    [[ 0.       , 0.],  0.       ],
    [[ 0.5*np.pi, 0.],  0.5*np.pi],
    [[     np.pi, 0.],      np.pi],
    [[ 1.5*np.pi, 0.],  1.5*np.pi],
    [[ 2.0*np.pi, 0.],  2.0*np.pi],
    [[ 2.5*np.pi, 0.],  2.5*np.pi],
    [[-0.5*np.pi, 0.5], -0.6141848493043787],
    [[ 0.       , 0.5],  0.                ],
    [[ 0.5*np.pi, 0.5],  0.6141848493043787],
    [[     np.pi, 0.5],      np.pi         ],
    [[ 1.5*np.pi, 0.5],  5.669000457875207 ],
    [[ 2.0*np.pi, 0.5],  2.0*np.pi         ],
    [[ 2.5*np.pi, 0.5],  6.897370156483965 ],
]

eccentric_anomaly_from_mean_anomaly = [
    # [[        mu,   e],      theta]
    [[-0.5*np.pi, 0.], -0.5*np.pi],    
    [[ 0.       , 0.],  0.       ],
    [[ 0.5*np.pi, 0.],  0.5*np.pi],
    [[     np.pi, 0.],      np.pi],
    [[ 1.5*np.pi, 0.],  1.5*np.pi],
    [[ 2.0*np.pi, 0.],  2.0*np.pi],
    [[ 2.5*np.pi, 0.],  2.5*np.pi],
    [[-0.5*np.pi, 0.5], -2.02097993808977 ],
    [[ 0.       , 0.5],  0.               ],
    [[ 0.5*np.pi, 0.5],  2.02097993808977 ],
    [[     np.pi, 0.5],      np.pi        ],
    [[ 1.5*np.pi, 0.5],  4.262205369089816],
    [[ 2.0*np.pi, 0.5],  2.0*np.pi        ],
    [[ 2.5*np.pi, 0.5],  8.304165245269356],
]

true_anomaly_from_mean_anomaly = [
    # [[        mu,   e],      theta]
    [[-0.5*np.pi, 0.], -0.5*np.pi],
    [[ 0.       , 0.],  0.       ],
    [[ 0.5*np.pi, 0.],  0.5*np.pi],
    [[     np.pi, 0.],      np.pi],
    [[ 1.5*np.pi, 0.],  1.5*np.pi],
    [[ 2.0*np.pi, 0.],  2.0*np.pi],
    [[ 2.5*np.pi, 0.],  2.5*np.pi],
    [[-0.5*np.pi, 0.5], -2.446560877968672 ],
    [[ 0.       , 0.5],  0.                ],
    [[ 0.5*np.pi, 0.5],  2.446560877968672 ],
    [[     np.pi, 0.5],      np.pi         ],
    [[ 1.5*np.pi, 0.5],  3.8366244292109135],
    [[ 2.0*np.pi, 0.5],  2.0*np.pi         ],
    [[ 2.5*np.pi, 0.5],  8.729746185148258 ],
]

primary_semimajor_axis_from_semimajor_axis = [
    # [[a, q], a_1]
    [[0.95128107, 0.2189373], 0.1708626924509661],
]

secondary_semimajor_axis_from_semimajor_axis = [
    # [[a, q], a_1]
    [[0.62427728, 0.16652542], 0.5351596024371248],
]

primary_semimajor_axis_from_secondary_semimajor_axis = [
    # [[a, q], a_1]
    [[0.11757373, 0.0908097], 0.010676835149181],
]

secondary_semimajor_axis_from_primary_semimajor_axis = [
    # [[a, q], a_1]
    [[0.99802898, 0.64952906], 1.5365424604712836],
]

initialization = [
    [[ 1., 1., -1.]], # Negative eccentricity
    [[ 1., 1.,  1.]], # Parabolic orbit
    [[ 1., 1.,  2.]], # Hyperbolic orbit
    [[ 0., 1.,  0.]], # Zero mass
    [[-1., 1.,  0.]], # Negative mass
]
