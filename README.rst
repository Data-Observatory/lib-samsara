========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |github-actions|
    * - package
      - | |commits-since|

..      | |codecov|
..      | |version| |wheel| |supported-versions| |supported-implementations|

.. |docs| image:: https://github.com/Data-Observatory/lib-samsara/actions/workflows/docs.yml/badge.svg
    :target: https://data-observatory.github.io/lib-samsara/
    :alt: Documentation Status

.. |github-actions| image:: https://github.com/Data-Observatory/lib-samsara/actions/workflows/test.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/Data-Observatory/lib-samsara/actions

.. |codecov| image:: https://codecov.io/gh/Data-Observatory/lib-samsara/branch/main/graphs/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://app.codecov.io/github/Data-Observatory/lib-samsara

.. |version| image:: https://img.shields.io/pypi/v/samsara.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/samsara

.. |wheel| image:: https://img.shields.io/pypi/wheel/samsara.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/samsara

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/samsara.svg
    :alt: Supported versions
    :target: https://pypi.org/project/samsara

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/samsara.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/samsara

.. |commits-since| image:: https://img.shields.io/github/commits-since/Data-Observatory/lib-samsara/v0.2.1.svg
    :alt: Commits since latest release
    :target: https://github.com/Data-Observatory/lib-samsara/compare/v0.2.1...main



.. end-badges

Package for the Satellite Alert and Monitoring System for Areas of Environmental Relevance (SAMSARA).

* Free software: GNU Lesser General Public License v3 (LGPLv3)

Installation
============

To install the package from the source (in an existing environment) use::

    git clone git@github.com:Data-Observatory/lib-samsara.git
    cd lib-samsara
    git checkout develop
    pip install .

You can also install from VCS::

    pip install git+ssh://git@github.com/Data-Observatory/lib-samsara

To install a specific branch (e.g. develop) from VCS use::

    pip install git+ssh://git@github.com/Data-Observatory/lib-samsara@develop

Documentation
=============


https://data-observatory.github.io/lib-samsara/


Development
===========

For development, download the repository content and install the package in editable mode with the
developer packages in a new environment::

    git clone git@github.com:Data-Observatory/lib-samsara.git
    cd lib-samsara
    git checkout develop
    # Create virtual environment
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e .[dev]
    # Check pre-commit is installed
    pre-commit install

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
