#!/usr/bin/env python
import re
from pathlib import Path

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    with Path(__file__).parent.joinpath(*names).open(encoding=kwargs.get('encoding', 'utf8')) as fh:
        return fh.read()


setup(
    name='samsara',
    use_scm_version={
        'local_scheme': 'dirty-tag',
        'write_to': 'src/samsara/_version.py',
        'fallback_version': '0.0.0',
    },
    license='LGPL-3.0-only',
    description='Package for the Satellite Alert and Monitoring System for Areas of Environmental Relevance (SAMSARA).',
    long_description='{}\n{}'.format(
        re.compile('^.. start-badges.*^.. end-badges', re.M | re.S).sub('', read('README.rst')),
        re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read('CHANGELOG.rst')),
    ),
    author='Data Observatory',
    author_email='devops-team@dataobservatory.net',
    url='https://github.com/Data-Observatory/lib-samsara',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[path.stem for path in Path('src').glob('*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)' 'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        # uncomment if you test on these interpreters:
        # "Programming Language :: Python :: Implementation :: IronPython",
        # "Programming Language :: Python :: Implementation :: Jython",
        # "Programming Language :: Python :: Implementation :: Stackless",
        'Topic :: Utilities',
        'Private :: Do Not Upload',
    ],
    project_urls={
        'Documentation': 'https://lib-samsara.readthedocs.io/',
        'Changelog': 'https://lib-samsara.readthedocs.io/en/latest/changelog.html',
        'Issue Tracker': 'https://github.com/Data-Observatory/lib-samsara/issues',
    },
    keywords=[
        # eg: "keyword1", "keyword2", "keyword3",
    ],
    python_requires='>=3.9',
    install_requires=[
        # eg: "aspectlib==1.1.1", "six>=1.7",
    ],
    extras_require={
        # eg:
        #   "rst": ["docutils>=0.11"],
        #   ":python_version=="2.6"": ["argparse"],
    },
    setup_requires=[
        'setuptools_scm>=3.3.1',
    ],
    entry_points={
        'console_scripts': [
            'samsara = samsara.cli:main',
        ]
    },
)
