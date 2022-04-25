# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys

import pt_lightning_sphinx_theme

SPHINX_MOCK_REQUIREMENTS = int(os.environ.get('SPHINX_MOCK_REQUIREMENTS', True))
_PATH_HERE = os.path.abspath(os.path.dirname(__file__))
_PATH_ROOT = os.path.join(_PATH_HERE, '..', '..')
sys.path.insert(0, os.path.abspath(_PATH_ROOT))

import lightning_transformers  # noqa: E402

# -- Project information -----------------------------------------------------

project = "Lightning Transformers"
copyright = "2020-2021, PyTorch Lightning"
author = "PyTorch Lightning"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    # 'sphinx.ext.todo',
    # 'sphinx.ext.coverage',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.imgmath',
    'recommonmark',
    # 'sphinx.ext.autosectionlabel',
    # 'nbsphinx',  # it seems some sphinx issue
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'sphinx_paramlinks',
    'sphinx_togglebutton',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["api", "readme.md"]  # TODO: ignored until fixed

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]

needs_sphinx = '3.4'

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pt_lightning_sphinx_theme"
html_theme_path = [pt_lightning_sphinx_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_theme_options = {
    "pytorch_project": lightning_transformers.__homepage__,
    "canonical_url": lightning_transformers.__homepage__,
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = []


def setup(app):
    # this is for hiding doctest decoration,
    # see: http://z4r.github.io/python/2011/12/02/hides-the-prompts-and-output/
    app.add_js_file('copybutton.js')


# Ignoring Third-party packages
# https://stackoverflow.com/questions/15889621/sphinx-how-to-exclude-imports-in-automodule
def _package_list_from_file(pfile):
    assert os.path.isfile(pfile)
    with open(pfile, 'r') as fp:
        lines = fp.readlines()
    list_pkgs = []
    for ln in lines:
        found = [ln.index(ch) for ch in list(',=<>#@') if ch in ln]
        pkg = ln[:min(found)] if found else ln
        if pkg.strip():
            list_pkgs.append(pkg.strip())
    return list_pkgs


# define mapping from PyPI names to python imports
PACKAGE_MAPPING = {
    'pytorch-lightning': 'pytorch_lightning',
    'scikit-learn': 'sklearn',
    'Pillow': 'PIL',
    'PyYAML': 'yaml',
    "hydra-core": "hydra",
    "rouge-score": "rouge_score",
}
MOCK_PACKAGES = ['omegaconf']
if SPHINX_MOCK_REQUIREMENTS:
    # mock also base packages when we are on RTD since we don't install them there
    MOCK_PACKAGES += _package_list_from_file(os.path.join(_PATH_ROOT, 'requirements.txt'))
# replace PyPI packages by importing ones
MOCK_PACKAGES = [PACKAGE_MAPPING.get(pkg, pkg) for pkg in MOCK_PACKAGES]

autodoc_mock_imports = MOCK_PACKAGES
