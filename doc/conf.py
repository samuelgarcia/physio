    # Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'physio'
copyright = '2023, Samuel Garcia'
author = 'Samuel Garcia'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx_gallery.gen_gallery',
    'numpydoc',
  ]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_static_path = ['_static']


# for sphinx gallery plugin

# from sphinx_gallery.sorting import ExplicitOrder
from sphinx_gallery.sorting import FileNameSortKey

sphinx_gallery_conf = {
    'only_warn_on_example_error': True,
    'examples_dirs': ['../examples/'],
    'gallery_dirs': ['examples', ],
    # 'subsection_order': ExplicitOrder([
    #                                    '../examples/example_01_overview.py',
    #                                    '../examples/example_02_respiration.py',
    #                                    ]),
    'within_subsection_order': FileNameSortKey,
    'filename_pattern' : '/example_',

    'ignore_pattern': '/generate_',
    'download_all_examples': False,
}

try:
    import sphinx_rtd_theme

    html_theme = "sphinx_rtd_theme"
    # html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
except ImportError:
    print("RTD theme not installed, using default")
    html_theme = 'alabaster'
