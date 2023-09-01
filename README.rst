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
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/lib-samsara/badge/?style=flat
    :target: https://lib-samsara.readthedocs.io/
    :alt: Documentation Status

.. |github-actions| image:: https://github.com/Data-Observatory/lib-samsara/actions/workflows/github-actions.yml/badge.svg
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

.. |commits-since| image:: https://img.shields.io/github/commits-since/Data-Observatory/lib-samsara/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/Data-Observatory/lib-samsara/compare/v0.0.0...main



.. end-badges

Package for the Satellite Alert and Monitoring System for Areas of Environmental Relevance (SAMSARA).

* Free software: GNU Lesser General Public License v3 (LGPLv3)

Installation
============

::

    pip install samsara

You can also install the in-development version with::

    pip install https://github.com/Data-Observatory/lib-samsara/archive/main.zip


Documentation
=============


https://lib-samsara.readthedocs.io/


Development
===========

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
