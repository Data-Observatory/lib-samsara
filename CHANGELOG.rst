
Changelog
=========

0.2.1 (2024-01-08)
------------------

Features

* Fix properties values when dealing with division by zero and nan values in statistics in
  :code:`stats.glcm`.

Pull request merged

* #5: Fix division values in stats.glcm

0.2.0 (2023-12-26)
------------------

Features

* Publish subpackages :code:`stats.glcm`.
* Refactor one filter type in :code:`filter` to allow filtering according to values at different
  breakpoints.
* Add a new statistic, refactor to Kernel usage for pixel neighborhood and change vectorization to
  Numba usage. All this in the :code:`stats.neighborhood` subpackage.
* Fix bug in :code:`pelt` subpackage. Avoid exception in cases where data for breakpoint prediction
  is not sufficient.
* Fix error that did not preserve all of the array coordinates and attributes in subpackages
  :code:`pelt` and :code:`filter`.

Pull request merged

* #4: Add GLCM and Fix Pelt


0.1.0 (2023-12-04)
------------------

Features

* Publish subpackages :code:`filter` and :code:`stats.neighborhood`.
* Refactor :code:`pelt` subpackage to return dataset intead of a dataarray. Also, now the dates are
  in UNIX time.

Pull request merged

* #3: Add filter and stats.neighborhood submodules


0.0.2 (2023-11-18)
------------------

Features

* Fix bug in :code:`pelt` subpackage. Number of returned arrays didn't match specified configuration.

Pull request merged

* #2: Fix Pelt xarray execution

0.0.1 (2023-11-15)
------------------

Features

* Publish subpackages :code:`kernel` and :code:`pelt`.

Pull request merged

* #1: Add Kernel and Pelt submodules
