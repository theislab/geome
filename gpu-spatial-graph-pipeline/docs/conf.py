"""Sphinx configuration."""
project = "Gpu Spatial Graph Pipeline"
author = "-"
copyright = "2022, -"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
