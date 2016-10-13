Setup
=====

Installation
------------

Installation is easy. After cloning the project onto your machine, simply use the `setup.py` file:

.. code-block:: bash

    git clone https://github.com/tgsmith61591/skutil.git
    cd skutil
    python setup.py install



Running tests
-------------

If you'd like to fork skutil and will be running some tests, your setup is a bit different. Rather than using the `install` arg, use `develop`. This creates a symlink in the local directory so that as you make changes, they are automatically reflected and you don't have to re-install every time. For more information on `develop` vs. `install`, see [this](http://stackoverflow.com/questions/19048732/python-setup-py-develop-vs-install) StackOverflow question. Note that after running setup with `develop`, you may have to uninstall before re-running with `install`. *If you are experiencing the dreaded* `no module named dqrsl` *issue and your GCC is up-to-date, it's likely a* `develop` *vs.* `install` *issue. Try uninstalling, clearing the egg from the local folder (or popping the local path from* `sys.path`*) and running setup with the* `install` *option.*

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
