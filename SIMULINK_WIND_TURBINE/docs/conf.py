# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'WindTurbine-Generator-predictive-maintenance'
copyright = '2025, BaquaAbdellah,HachimBoua'
author = 'BaquaAbdellah,HachimBoua'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
# Configuration file for the Sphinx documentation builder.

import os
import sys
from datetime import datetime

# -- Path setup ---------------------------------------------------------------
# Add project root to sys.path to allow autodoc to find modules
sys.path.insert(0, os.path.abspath(".."))

# -- Project information ------------------------------------------------------
project = 'Wind Turbine Predictive Maintenance'
author = 'Your Name or Team'
copyright = f'{datetime.now().year}, {author}'
release = '1.0.0'

# -- General configuration ----------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',           # Include documentation from docstrings
    'sphinx.ext.napoleon',          # Support for Google and NumPy style docstrings
    'sphinx.ext.viewcode',          # Add links to highlighted source code
    'sphinx.ext.todo',              # Support for todo directives
    'myst_parser',                  # Parse Markdown files
]

# Enable TODOs in the documentation
todo_include_todos = True

# Source file types
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Master document
master_doc = 'index'

# Language
language = 'en'

# Exclude patterns
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- HTML output --------------------------------------------------------------
html_theme = 'furo'  # You can change this to 'sphinx_rtd_theme' or another theme
# html_theme = 'sphinx_rtd_theme'

# Static files (e.g., custom CSS)
# html_static_path = ['_static']

# -- Options for autodoc ------------------------------------------------------
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# -- Napoleon settings --------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Options for HTML title ---------------------------------------------------
html_title = project
