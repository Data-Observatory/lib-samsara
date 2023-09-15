import traceback

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.ifconfig',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
]
source_suffix = '.rst'
master_doc = 'index'
project = 'SAMSARA'
year = '2023-2024'
author = 'Data Observatory'
copyright = f'{year}, {author}'
try:
    from pkg_resources import get_distribution

    version = release = get_distribution('samsara').version
except Exception:
    traceback.print_exc()
    version = release = '0.0.0'

pygments_style = 'trac'
templates_path = ['.']
extlinks = {
    'issue': ('https://github.com/Data-Observatory/lib-samsara/issues/%s', '#'),
    'pr': ('https://github.com/Data-Observatory/lib-samsara/pull/%s', 'PR #'),
}
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    'githuburl': 'https://github.com/Data-Observatory/lib-samsara/',
}

html_use_smartypants = True
html_last_updated_fmt = '%b %d, %Y'
html_split_index = False
html_sidebars = {
    '**': ['searchbox.html', 'globaltoc.html', 'sourcelink.html'],
}
html_short_title = f'{project}-{version}'

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False
