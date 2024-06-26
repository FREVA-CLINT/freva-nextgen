# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
from datetime import date

import freva_client
from freva_client import __version__
from freva_client.auth import getpass

project = "Freva Databrowser"
copyright = f"{date.today().year}, DKRZ"
author = "DKRZ"
release = __version__


def getpass_mock(msg: str, result: str = "janedoe123") -> str:
    """Mock the getpass method."""
    print(f"{msg} *****")
    return result


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_code_tabs",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "sphinxcontrib.httpdomain",
    "sphinx_execute_code",
]

html_static_path = ["_static"]
html_theme = "pydata_sphinx_theme"
html_logo = os.path.join(html_static_path[0], "logo.png")
templates_path = ["_templates"]
exclude_patterns = []
html_favicon = html_logo
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/FREVA-CLINT/freva-nextgen/freva-client",
            "icon": "fa-brands fa-github",
        }
    ],
    "navigation_with_keys": False,
    "show_toc_level": 4,
    "collapse_navigation": False,
    "navigation_depth": 4,
    "navbar_align": "left",
    "show_nav_level": 4,
    "navigation_depth": 4,
    "navbar_center": ["navbar-nav"],
    "secondary_sidebar_items": ["page-toc"],
}

html_context = {
    "github_user": "FREVA-CLINT",
    "github_repo": "freva-nextgen",
    "github_version": "main",
    "doc_path": "docs",
}
html_sidebars = {"**": ["search-field", "sidebar-nav-bs"]}

# -- Options for autosummary/autodoc output ------------------------------------
autosummary_generate = True
# autodoc_typehints = "description"
# autodoc_class_signature = "separated"
# autodoc_member_order = "groupwise"


# -- Options for autoapi -------------------------------------------------------
autoapi_type = "python"
autoapi_dirs = ["../src/databrowser"]
autoapi_keep_files = True
autoapi_root = "api"
autoapi_member_order = "groupwise"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# -- MyST options ------------------------------------------------------------

# This allows us to use ::: to denote directives, useful for admonitions
myst_enable_extensions = ["colon_fence", "linkify", "substitution"]
myst_heading_anchors = 2
myst_substitutions = {"rtd": "[Read the Docs](https://readthedocs.org/)"}

# ReadTheDocs has its own way of generating sitemaps, etc.
if not os.environ.get("READTHEDOCS"):
    extensions += ["sphinx_sitemap"]

    html_baseurl = os.environ.get("SITEMAP_URL_BASE", "http://127.0.0.1:8000/")
    sitemap_locales = [None]
    sitemap_url_scheme = "{link}"

# specifying the natural language populates some key tags
language = "en"
freva_client.auth.getpass = getpass_mock
getpass = getpass_mock
