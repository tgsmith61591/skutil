.. BADGES
   ======
.. image:: https://travis-ci.org/tgsmith61591/skutil.svg?branch=master
   :target: https://travis-ci.org/tgsmith61591/skutil
.. image:: https://coveralls.io/repos/github/tgsmith61591/skutil/badge.svg?branch=master
   :target: https://coveralls.io/github/tgsmith61591/skutil?branch=master
.. image:: https://img.shields.io/badge/python-2.7-blue.svg
   :target: https://img.shields.io/badge/python-2.7-blue.svg
.. image:: https://img.shields.io/github/release/tgsmith61591/skutil.svg
   :target: https://img.shields.io/github/release/tgsmith61591/skutil
.. image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :target: https://github.com/tgsmith61591/skutil/blob/master/LICENSEl
.. image:: https://img.shields.io/twitter/follow/TayGriffinSmith.svg?style=social
   :target: https://twitter.com/TayGriffinSmith


.. raw:: html
   <br/><br/>


.. image:: https://github.com/tgsmith61591/skutil/blob/master/doc/images/h2o-sklearn.png
   :target: https://github.com/tgsmith61591/skutil/blob/master/doc/images/h2o-sklearn.png


scikit-util
===========

What began as a modest, succinct set of `sklearn <https://github.com/scikit-learn/scikit-learn>`_ extension classes and utilities (as well as implementations of preprocessors from R packages like `caret <https://github.com/topepo/caret>`_) grew to bridge functionality between sklearn and `H2O <https://github.com/h2oai/h2o-3>`_.  Now, scikit-util (skutil) brings the best of both worlds to H2O and sklearn, delivering an easy transition into the world of distributed computing that H2O offers, while providing the same, familiar interface that sklearn users have come to know and love. View the `documentation here <https://tgsmith61591.github.io/skutil>`_.


Pre-installation:
-----------------

Skutil adapts code from several R packages, and thus depends on the ability to compile Fortran code using `gcc`. For different platforms, there are different ways to install `gcc` (the easiest, of course, being `Homebrew <http://brew.sh/>`_):
  
  - **Mac OS** (**note**: this can take a while):
  
``$ brew install gcc``

There is a bug in some setups that will still cause issues in symlinking the `gcc` files via homebrew. If this is the case, the following line should clear things up:


``$ brew link --overwrite gcc``

  - **Linux**:

``$ sudo apt-get install gcc``

  - For Windows, follow `this tutorial <http://www.preshing.com/20141108/how-to-install-the-latest-gcc-on-windows/>`_

Installation:
-------------

Installation is easy. After cloning the project onto your machine and installing the required dependencies, simply use the `setup.py` file:

.. code-block:: bash

    $ git clone https://github.com/tgsmith61591/skutil.git
    $ cd skutil
    $ python setup.py install

Testing:
--------

After installation, you can launch the test suite from outside the source directory (you will need to have the `nose` package installed):

``$ nosetests -v skutil``

Examples:
---------

  - See the `example ipython notebooks <https://github.com/tgsmith61591/skutil/tree/master/doc/examples>`_

