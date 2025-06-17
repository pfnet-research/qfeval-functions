# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'qfeval-functions'
copyright = '2025, qfeval-functions Team'
author = 'qfeval-functions Team'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'functions/_template.rst']

# -- Autodoc configuration --------------------------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# -- Autosummary configuration -----------------------------------------------
autosummary_generate = True
autosummary_generate_overwrite = True

# Configure doctest to test docstrings in autodoc
doctest_test_doctest_blocks = 'all'
# Test doctests in function docstrings 
autodoc_docstring_signature = True

# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Doctest configuration --------------------------------------------------
doctest_global_setup = """
import torch
import qfeval_functions.functions as QF
"""

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']