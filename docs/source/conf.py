# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# abs_path = os.getcwd()
# print(abs_path)
# while True:
#     basename = os.path.basename(abs_path)
#     if basename.lower() == 'multimodal-toolkit':
#         break
#     abs_path = os.path.dirname(abs_path)
# print(abs_path)
# for x in os.listdir(abs_path):
#     print(x)
# sys.path.insert(0, abs_path)
# sys.path.insert(0, os.path.join(abs_path, 'mutlimodal'))
sys.path.append(os.path.join(os.path.dirname(__name__), "../"))
sys.path.append(os.path.join(os.path.dirname(__name__), "../../"))

# -- Project information -----------------------------------------------------

project = 'multimodal toolkit'
copyright = '2020, Ken'
author = 'Ken'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinxcontrib.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'recommonmark',
    'sphinx_markdown_tables',
    'sphinx.ext.githubpages'
]
    # Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

autosummary_generate = True

source_suffix = ['.rst', '.md']
master_doc = 'index'
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

add_module_names = False


def setup(app):
    def skip(app, what, name, obj, skip, options):
        members = [
            '__init__',
            '__repr__',
            '__weakref__',
            '__dict__',
            '__module__',
        ]
        return True if name in members else skip

    app.connect('autodoc-skip-member', skip)