[![Build status](https://travis-ci.org/tgsmith61591/skutil.svg?branch=master)](https://travis-ci.org/tgsmith61591/skutil)
[![Coverage Status](https://coveralls.io/repos/github/tgsmith61591/skutil/badge.svg?branch=master)](https://coveralls.io/github/tgsmith61591/skutil?branch=master)
![Supported versions](https://img.shields.io/badge/python-2.7-blue.svg) 


<br/><br/>
![h2o-sklearn](doc/images/h2o-sklearn.png)


# scikit-util
What began as a modest, succinct set of [sklearn](https://github.com/scikit-learn/scikit-learn) extension classes and utilities (as well as implementations of preprocessors from R packages like [caret](https://github.com/topepo/caret)) grew to bridge functionality between sklearn and [H2O](https://github.com/h2oai/h2o-3).  Now, scikit-util (skutil) brings the best of both worlds to H2O and sklearn, delivering an easy transition into the world of distributed computing that H2O offers, while providing the same, familiar interface that sklearn users have come to know and love. __View the [documentation here](https://tgsmith61591.github.io/skutil)__



### Pre-installation
Skutil adapts code from several R packages, and thus depends on the ability to compile Fortran code using `gcc`. For different platforms, there are different ways to install `gcc` (the easiest, of course, being [Homebrew](http://brew.sh/)):
  - __Mac OS__ (__note__: this can take a while):
```bash
brew install gcc
```

There is a bug in some setups that will still cause issues in symlinking the `gcc` files via homebrew. If this is the case, the following line should clear things up:
```bash
brew link --overwrite gcc
```

  - __Linux__:
```bash
sudo apt-get install gcc
```

  - For Windows, follow [this tutorial](http://www.preshing.com/20141108/how-to-install-the-latest-gcc-on-windows/)




### Installation:

Installation is easy. After cloning the project onto your machine and installing the required dependencies, simply use the `setup.py` file:

```bash
git clone https://github.com/tgsmith61591/skutil.git
cd skutil
python setup.py install
```


### Installing for ongoing development:

If you'd like to fork skutil to contribute to the codebase and intend to run some tests, your setup is a bit different. Rather than using the `install` arg, use `develop`. This creates a symlink in the local directory so that as you make changes, they are automatically reflected and you don't have to re-install every time. For more information on `develop` vs. `install`, see [this](http://stackoverflow.com/questions/19048732/python-setup-py-develop-vs-install) StackOverflow question. Note that after running setup with `develop`, you may have to uninstall before re-running with `install`. *If you are experiencing the dreaded* `no module named dqrsl` *issue and your GCC is up-to-date, it's likely a* `develop` *vs.* `install` *issue. Try uninstalling, clearing the egg from the local folder (or popping the local path from* `sys.path`*) and running setup with the* `install` *option.*

```bash
git clone https://github.com/tgsmith61591/skutil.git
cd skutil
python setup.py develop
nosetests
```


#### Examples:
  - See the [example ipython notebooks](https://github.com/tgsmith61591/skutil/tree/master/doc/examples)

