[![Build status](https://travis-ci.org/tgsmith61591/skutil.svg?branch=master)](https://travis-ci.org/tgsmith61591/skutil)
[![Coverage Status](https://coveralls.io/repos/github/tgsmith61591/skutil/badge.svg?branch=master)](https://coveralls.io/github/tgsmith61591/skutil?branch=master)
![Supported versions](https://img.shields.io/badge/python-2.7-blue.svg) 

**Note: skutil has been deprecated and will no longer be supported. See its new sister project**: [skoot](https://github.com/tgsmith61591/skoot)

<br/><br/>
![h2o-sklearn](doc/images/h2o-sklearn.png)


# scikit-util
What began as a modest, succinct set of [sklearn](https://github.com/scikit-learn/scikit-learn) extension classes and utilities (as well as implementations of preprocessors from R packages like [caret](https://github.com/topepo/caret)) grew to bridge functionality between sklearn and [H2O](https://github.com/h2oai/h2o-3).  Now, scikit-util (skutil) brings the best of both worlds to H2O and sklearn, delivering an easy transition into the world of distributed computing that H2O offers, while providing the same, familiar interface that sklearn users have come to know and love. __View the [documentation here](https://tgsmith61591.github.io/skutil)__



### Pre-installation
Skutil adapts code from several R packages, and thus depends on the ability to compile Fortran code using `gcc`. For different platforms, there are different ways to install `gcc` (the easiest, of course, being [Homebrew](http://brew.sh/)):
  - __Mac OS__ (__note__: this can take a while):
```bash
$ brew install gcc
```

There is a bug in some setups that will still cause issues in symlinking the `gcc` files via homebrew. If this is the case, the following line should clear things up:
```bash
$ brew link --overwrite gcc
```

  - __Linux__:
```bash
$ sudo apt-get install gcc
```

  - For Windows, follow [this tutorial](http://www.preshing.com/20141108/how-to-install-the-latest-gcc-on-windows/)




### Installation:

Installation is easy. After cloning the project onto your machine and installing the required dependencies, simply use the `setup.py` file:

```bash
$ git clone https://github.com/tgsmith61591/skutil.git
$ cd skutil
$ python setup.py install
```

### Testing

After installation, you can launch the test suite from outside the source directory (you will need to have the `nose` package installed):

```bash
$ nosetests -v skutil
```

#### Examples:
  - See the [example ipython notebooks](https://github.com/tgsmith61591/skutil/tree/master/doc/examples)

