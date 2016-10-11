Setup
=====

Installation
------------

.. code-block:: bash

    git clone https://github.com/tgsmith61591/skutil.git
    cd skutil
    python setup.py install


Running Tests
-------------

.. code-block:: bash

    git clone https://github.com/tgsmith61591/skutil.git
    cd skutil
    python setup.py develop
    nosetests

Troubleshooting Installation Issues
-----------------------------------

Skutil depends on the ability to compile Fortran code. For different platforms, there are different ways to install ``gcc``:

- Mac OS (this may take a while):

.. code-block:: bash

    brew install gcc


There is a bug in some setups that will still cause issues in symlinking the ``gcc`` files via homebrew.
If this is the case, the following line should clear things up:

.. code-block:: bash

    brew link --overwrite gcc

- Linux:

.. code-block:: bash

    sudo apt-get install gcc

- For Windows, follow `this tutorial <http://www.preshing.com/20141108/how-to-install-the-latest-gcc-on-windows/>`_
