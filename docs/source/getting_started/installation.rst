.. _installation:

************
Installation
************

You can install Dyad from the source distribution, available at
the Dyad repository
(`https://github.com/AmeryGration/dyad`__). Make a local copy of
the repository and navigate to its top level. Then run the following command.

.. sourcecode:: bash

   $ python -m pip install .

Dyad requires Python version 3.10 or greater.

To verify your installation you can run Dyad's test suite as follows.

.. sourcecode:: bash

   $ python -m unittest discover ./tests

__ https://github.com/AmeryGration/dyad

Conda
=====

You may wish to install Dyad within a virtual environment. If you use Conda to manage your environments then you can use the file ``environment.yml`` to create an environment called 'dyad'.

.. sourcecode:: bash

   $ conda env create -f environment.yml
   $ conda activate dyad

Having created an environment in this way you can now install Dyad by following the previous instructions.
