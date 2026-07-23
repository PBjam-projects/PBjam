"""Sphinx configuration for the PBjam documentation."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    release = package_version("pbjam")
except PackageNotFoundError:
    release = "0+unknown"

project = "PBjam"
author = "The PBjam developers"
copyright = "2019–2026, The PBjam developers"
version = release.split("+", maxsplit=1)[0]

needs_sphinx = "7.2"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_preserve_defaults = True

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

nbsphinx_allow_errors = False
nbsphinx_execute = "never"

exclude_patterns = [
    "_build",
    "**/.ipynb_checkpoints",
]
templates_path = ["_templates"]
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "nbsphinx",
}
root_doc = "index"
language = "en"

html_theme = "sphinx_rtd_theme"
html_title = f"PBjam {version} documentation"
html_static_path = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "astropy": ("https://docs.astropy.org/en/stable", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

nitpicky = False

latex_documents = [
    (root_doc, "PBjam.tex", "PBjam Documentation", author, "manual"),
]
man_pages = [
    (root_doc, "pbjam", "PBjam Documentation", [author], 1),
]
texinfo_documents = [
    (
        root_doc,
        "PBjam",
        "PBjam Documentation",
        author,
        "PBjam",
        "Automated mode identification and peakbagging for solar-like oscillators.",
        "Science",
    )
]
epub_title = project
epub_exclude_files = ["search.html"]
