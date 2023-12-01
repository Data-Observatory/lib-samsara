import traceback

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.ifconfig",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "autoapi.extension",
    "nbsphinx",
]

source_suffix = ".rst"
master_doc = "index"
project = "SAMSARA"
year = "2023-2024"
author = "Data Observatory"
copyright = f"{year}, {author}"

try:
    from pkg_resources import get_distribution

    version = release = get_distribution("samsara").version
except Exception:
    traceback.print_exc()
    version = release = "0.1.0"

pygments_style = "trac"
templates_path = ["."]
extlinks = {
    "issue": ("https://github.com/Data-Observatory/lib-samsara/issues/%s", "#"),
    "pr": ("https://github.com/Data-Observatory/lib-samsara/pull/%s", "PR #"),
}
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/Data-Observatory/lib-samsara/",
}

html_use_smartypants = True
html_last_updated_fmt = "%b %d, %Y"
html_split_index = False
html_sidebars = {
    "**": ["searchbox.html", "globaltoc.html", "sourcelink.html"],
}
html_short_title = f"{project}-{version}"

autoapi_type = "python"
autoapi_dirs = ["../src/samsara"]
autoapi_file_patterns = ["*.py"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
