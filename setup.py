#!/usr/bin/env python
import re
from pathlib import Path

from setuptools import find_packages, setup


def read(*names, **kwargs):
    with Path(__file__).parent.joinpath(*names).open(
        encoding=kwargs.get("encoding", "utf8")
    ) as fh:
        return fh.read()


setup(
    name="samsara",
    use_scm_version={
        "local_scheme": "node-and-date",
        "write_to": "src/samsara/_version.py",
        "fallback_version": "0.0.2",
    },
    license="LGPL-3.0-only",
    description="Package for the Satellite Alert and Monitoring System for Areas of Environmental Relevance (SAMSARA).",
    long_description="{}\n{}".format(
        re.compile("^.. start-badges.*^.. end-badges", re.M | re.S).sub(
            "", read("README.rst")
        ),
        re.sub(":[a-z]+:`~?(.*?)`", r"``\1``", read("CHANGELOG.rst")),
    ),
    author="Data Observatory",
    author_email="devops-team@dataobservatory.net",
    url="https://github.com/Data-Observatory/lib-samsara",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[path.stem for path in Path("src").glob("*.py")],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)"
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        # uncomment if you test on these interpreters:
        # "Programming Language :: Python :: Implementation :: CPython",
        # "Programming Language :: Python :: Implementation :: PyPy",
        # "Programming Language :: Python :: Implementation :: IronPython",
        # "Programming Language :: Python :: Implementation :: Jython",
        # "Programming Language :: Python :: Implementation :: Stackless",
        "Topic :: Utilities",
        "Private :: Do Not Upload",
    ],
    project_urls={
        "Documentation": "https://data-observatory.github.io/lib-samsara/",
        "Changelog": "https://data-observatory.github.io/lib-samsara/changelog.html",
        "Issue Tracker": "https://github.com/Data-Observatory/lib-samsara/issues",
    },
    keywords=[
        # eg: "keyword1", "keyword2", "keyword3",
    ],
    python_requires=">=3.9",
    install_requires=[
        "dask>=2023.10.0",
        "numba>=0.58.1",
        "rioxarray>=0.15.0",
        "ruptures>=1.1.8",
        "scikit-image>=0.22.0",
        "scikit-learn>=1.3.2",
        "xarray>=2023.10.1",
    ],
    extras_require={
        "dev": [
            "black>=23.9.1",
            "isort>=5.12.0",
            "pytest>=7.4",
            "pytest-cov>=4.1.0",
            "pre-commit>=3.4.0",
            "ruff>=0.0.289",
        ]
    },
    setup_requires=[
        "setuptools_scm>=3.3.1",
    ],
    entry_points={
        "console_scripts": [
            "samsara = samsara.cli:main",
        ]
    },
)
