import os
import sys
import math
import matplotlib as mpl
import matplotlib.pyplot as plt

import dyad

sys.path.insert(0, os.path.abspath("../../dyad"))

project = "Dyad"
copyright = "2024, Amery Gration"
author = "Amery Gration"
release = "0.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    # "sphinx_design",
    # "matplotlib.sphinxext.plot_directive",
    "sphinx_rtd_theme"
]

templates_path = ["_templates"]
html_static_path = ["_static"]

# exclude_patterns = ["./data/"]

intersphinx_mapping = {
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# html_theme = "alabaster"
html_theme = "pydata_sphinx_theme"

# mpl.use("agg")
# plt.ioff()

# ############################################################################
# # Matplotlib plot_directive options
# ############################################################################

# plot_pre_code = """
# import warnings
# import numpy as np

# np.random.seed(123)

# """

# plot_include_source = True
# plot_formats = [("png", 96)]
# plot_html_show_formats = False
# plot_html_show_source_link = False

# phi = (math.sqrt(5) + 1)/2

# font_size = 13.*72./96.  # 13 px

# plot_rcparams = {
#     "font.size": font_size,
#     "axes.titlesize": font_size,
#     "axes.labelsize": font_size,
#     "xtick.labelsize": font_size,
#     "ytick.labelsize": font_size,
#     "legend.fontsize": font_size,
#     "figure.figsize": (3*phi, 3),
#     "figure.subplot.bottom": 0.2,
#     "figure.subplot.left": 0.2,
#     "figure.subplot.right": 0.9,
#     "figure.subplot.top": 0.85,
#     "figure.subplot.wspace": 0.4,
#     "text.usetex": False,
# }
