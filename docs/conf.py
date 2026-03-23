# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from importlib.metadata import version as _get_version, PackageNotFoundError
from pathlib import Path

# Add src/ to sys.path so autodoc can import raspy without installing it.
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

# -- Project information -----------------------------------------------------
project = "raspy"
copyright = "2025, Gyan Basyal and WEST Consultants, Inc."
author = "Gyan Basyal and WEST Consultants, Inc."

try:
    release = _get_version("raspy")
except PackageNotFoundError:
    release = "0.0.0.dev0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",        # Google/NumPy docstring styles
    "sphinx.ext.viewcode",        # [source] links in API docs
    "sphinx.ext.intersphinx",     # cross-link to numpy, pandas, etc.
    "sphinx_autodoc_typehints",   # render type hints in descriptions
    "myst_parser",                # parse .md files as pages
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "h5py": ("https://docs.h5py.org/en/stable", None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = "raspy"
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
