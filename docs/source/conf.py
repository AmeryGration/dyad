import math
import os
import re
import sys
import warnings
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy._lib.uarray as ua
import numpydoc.docscrape as np_docscrape
import doctest
import sphinx_rtd_theme

from os.path import relpath, dirname
from docutils import nodes
from docutils.parsers.rst import Directive
from intersphinx_registry import get_intersphinx_mapping
from numpydoc.docscrape_sphinx import SphinxDocString
from sphinx.util import inspect
from scipy._lib._util import _rng_html_rewrite
# Workaround for sphinx-doc/sphinx#6573
# ua._Function should not be treated as an attribute
from scipy.stats._distn_infrastructure import rv_generic
from scipy.stats._multivariate import multi_rv_generic

old_isdesc = inspect.isdescriptor
inspect.isdescriptor = (lambda obj: old_isdesc(obj)
                        and not isinstance(obj, ua._Function))

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

########################################################################
# General configuration
########################################################################

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx_copybutton",
    "sphinx_design",
    "numpydoc",
    "matplotlib.sphinxext.plot_directive",
]

html_logo = "dyad_logo_white.png"

# Do some matplotlib config in case users have a matplotlibrc that will break
# things
matplotlib.use("agg")
plt.ioff()

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

numfig = True

# The main toctree document.
master_doc = "index"

# General substitutions.
sys.path.insert(0, os.path.abspath("../../dyad"))

project = "Dyad"
copyright = "2026, Amery Gration"
author = "Amery Gration"
version = "0.0.0"
release = "0.0.0"

if os.environ.get("CIRCLE_JOB", False) and \
        os.environ.get("CIRCLE_BRANCH", "") != "main":
    version = os.environ["CIRCLE_BRANCH"]
    release = version

print(f"{project} (VERSION {version})")

today_fmt = "%B %d, %Y"

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "autolink"

# If true, "()" will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

doctest_optionflags = [doctest.ELLIPSIS]

########################################################################
# HTML output
########################################################################

html_theme = "sphinx_rtd_theme"

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True

########################################################################
# Intersphinx configuration
########################################################################
intersphinx_mapping = get_intersphinx_mapping(
    packages={
        "python",
        "numpy",
        "scipy",
        "neps",
        "matplotlib",
        "asv",
        "statsmodels",
        "mpmath"
    }
)

########################################################################
# Numpy extensions
########################################################################

# If we want to do a phantom import from an XML file for all autodocs
phantom_import_file = "dump.xml"

# Generate plots for example sections
numpydoc_use_plots = True
np_docscrape.ClassDoc.extra_public_methods = [  # should match class.rst
    "__call__", "__mul__", "__getitem__", "__len__",
]

########################################################################
# Autosummary
########################################################################

# autosummary_generate = True

# # maps functions with a name same as a class name that is indistinguishable
# # Ex: scipy.signal.czt and scipy.signal.CZT or scipy.odr.odr and scipy.odr.ODR
# # Otherwise, the stubs are overwritten when the name is same for
# # OS (like MacOS) which has a filesystem that ignores the case
# # See https://github.com/sphinx-doc/sphinx/pull/7927
# autosummary_filename_map = {
#     "scipy.odr.odr": "odr-function",
#     "scipy.signal.czt": "czt-function",
#     "scipy.signal.ShortTimeFFT.t": "scipy.signal.ShortTimeFFT.t.lower",
# }

########################################################################
# Autodoc
########################################################################

# autodoc_default_options = {
#     "inherited-members": None,
# }
# autodoc_typehints = "none"

########################################################################
# Matplotlib plot_directive options
########################################################################

plot_pre_code = (
"""
# import warnings
# for key in (
#     'interp2d` is deprecated',
#     'scipy.misc',
#     '`kurtosistest` p-value may be',
#     ):
# warnings.filterwarnings(action='ignore', message='.*' + key + '.*')

import numpy as np
np.random.seed(123)

"""
)

plot_include_source = True
plot_formats = [("png", 96)]
plot_html_show_formats = False
plot_html_show_source_link = False

phi = (math.sqrt(5) + 1)/2

font_size = 13*72/96.0  # 13 px

plot_rcparams = {
    "font.size": font_size,
    "axes.titlesize": font_size,
    "axes.labelsize": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "legend.fontsize": font_size,
    "figure.figsize": (3*phi, 3),
    "figure.subplot.bottom": 0.2,
    "figure.subplot.left": 0.2,
    "figure.subplot.right": 0.9,
    "figure.subplot.top": 0.85,
    "figure.subplot.wspace": 0.4,
    "text.usetex": False,
}


