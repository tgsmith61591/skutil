[![Build status](https://travis-ci.org/tgsmith61591/skutil.svg?branch=master)](https://travis-ci.org/tgsmith61591/skutil)
[![Coverage Status](https://coveralls.io/repos/github/tgsmith61591/skutil/badge.svg?branch=master)](https://coveralls.io/github/tgsmith61591/skutil?branch=master)
![Supported versions](https://img.shields.io/badge/python-2.7-blue.svg) 


### scikit-util
A succinct set of [sklearn](https://github.com/scikit-learn/scikit-learn) extension classes.  


#### Installation:
```bash
git clone https://github.com/tgsmith61591/skutil.git
cd skutil
python easy_setup.py

# NOTE: easy_setup.py takes care of the following:
# python setup.py clean --all
# python setup.py install
```

#### Troubleshooting Installation Issues
Skutil depends on the ability to compile Fortran code. For different platforms, there are different ways to install `gcc`:
  - Mac OS (__note__: this can take a while):
```bash
brew install gcc
```

  - Linux:
```bash
sudo apt-get install gcc
```

  - For Windows, follow [this tutorial](http://www.preshing.com/20141108/how-to-install-the-latest-gcc-on-windows/)

#### Examples:
  - See the [wiki](https://github.com/tgsmith61591/skutil/wiki)
  - See the [example ipython notebooks](https://github.com/tgsmith61591/skutil/tree/master/doc/examples)

