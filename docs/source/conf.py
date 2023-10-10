# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'lhe2root'
copyright = '2023, HexUtils'
author = 'HexUtils'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    ]

autoclass_content = "init"

#Napoleon Settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = {
    'array-like': ':term:`array-like <array_like>`',
    'array_like': ':term:`array_like`',
}


autodoc_typehints = "description"

templates_path = ['_templates']
exclude_patterns = []

#Getting it to find the modules
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
# html_theme = 'classic'
html_static_path = ['_static']
html_theme_options = {}


html_sidebars = { '**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html', 'index.html', 'py-modindex.html'] }

intersphinx_mapping = {'Python 3': ('https://docs.python.org/3', None),
'matplotlib': ('https://matplotlib.org/stable/', None),
'numpy': ('https://numpy.org/doc/stable/', None),
'Sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}


# html_sidebars = {
#         '**': sidebar_list
#     }


