# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import datetime
import os
import sys

# get release from environment variable
version = os.environ.get("VERSION", "")
if not version:
    print("Error: VERSION environment variable not set.")
    sys.exit(1)

sys.path.insert(0, "release-code")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "DeepTrack2"
copyright = f"{datetime.now().year}, The DeepTrack2 Team"
author = "The DeepTrack2 Team"
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_automodapi.automodapi", 
    "sphinx.ext.githubpages",
]
numpydoc_show_class_members = False
automodapi_inheritance_diagram = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
    "switcher": {
        "json_url": "https://DeepTrackAI.github.io/DeepTrack2/latest/_static/switcher.json",
        "version_match": version,
    },
    "navbar_end": [
        "version-switcher",
        "navbar-icon-links",
    ],
}
