from setuptools import setup
from pathlib import Path

# See here: https://github.com/phobson/paramnormal/blob/main/setup.py
# See here: https://setuptools.pypa.io/en/latest/
# List of keywords:
# https://setuptools.pypa.io/en/latest/references/keywords.html

# For creating package data at build time:
# https://digip.org/blog/2011/01/generating-data-files-in-setup.py.html,
# https://stackoverflow.com/questions/55648982/generate-data-file-at-install-
# time

description = (
    "Dyad is a two-body dynamics and binary-star statistics package for "
    "astrophysicists."
)
long_description = (Path(__file__).parent/"README.md").read_text()

setup(
    name="dyad",
    version="0.0.0",
    description=description, 
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Amery Gration",
    author_email="amerygration@proton.me",
    url="https://github.com/AmeryGration/dyad",
    packages=["dyad", "dyad.constants", "dyad.stats"],
    license_file="COPYING",
    keywords=[
        # List of keywords:
        # https://packaging.python.org/en/latest/specifications/
        # core-metadata/#core-metadata-keywords
        "binary", "star", "astrophysics", "stellar", "population",
        "synthesis", "orbit", "keplerian", "dynamics", "astrodynamics",
        "celestial", "mechanics"
    ],
    classifiers=[
        # List of classifiers here: https://pypi.org/classifiers/
        "Development Status :: 2 – Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.10",
    ],
    # platforms="",
    # cmdclass="",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "numpydoc",
        "intersphinx_registry",
        "sphinx_copybutton",
        "parameterized",
    ],
    include_package_data=True,
    python_requires=">=3.10",
)
