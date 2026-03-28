# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add the parent directory to the path so that we can import gaint
sys.path.insert(0, os.path.abspath('../../'))

# Project information
project = 'GaInt'
copyright = '2026, Yilin Zhao'
author = 'Yilin Zhao'
release = '0.1.0'
version = '0.1.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',      # Math formula rendering
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_nb',                 # Jupyter notebook and Markdown support
    'sphinxcontrib.apidoc'
]

# sphinxcontrib-apidoc configuration
apidoc_module_dir = '../../gaint'        # Source code directory
apidoc_output_dir = 'api'                  # Output directory for .rst files
apidoc_excluded_paths = ['tests']          # Directories to exclude
apidoc_separate_modules = True             # Separate page for each module

# MyST configuration - enable $$ math formulas
myst_enable_extensions = [
    "dollarmath",      # Enable $$ and $ math formulas
    "amsmath",         # Enable AMS math environments
]

# Notebook execution settings
nb_execution_mode = "auto"      # Execute notebooks automatically
nb_execution_timeout = 300      # Timeout in seconds

# Patterns to exclude from build
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# HTML theme
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'logo_only': True,
    'prev_next_buttons_location': 'bottom',
    'style_nav_header_background': '#2980B9',
}

# Static files
html_logo = './_static/logo.png'
html_favicon = './_static/logo.ico'
html_static_path = ['_static']

# Napoleon settings for docstring parsing
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Autodoc settings
autodoc_typehints = 'description'
autodoc_member_order = 'bysource'

# Intersphinx mapping for cross-referencing
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# MathJax configuration - supports $$ and $ formulas
mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
        'processEscapes': True,
    },
}