[![Build status](https://travis-ci.org/tgsmith61591/skutil.svg?branch=master)](https://travis-ci.org/tgsmith61591/skutil)
[![Coverage Status](https://coveralls.io/repos/github/tgsmith61591/skutil/badge.svg?branch=master)](https://coveralls.io/github/tgsmith61591/skutil?branch=master)
![Supported versions](https://img.shields.io/badge/python-2.7-blue.svg) 


<br/><br/>
![h2o-sklearn](doc/images/h2o-sklearn.png)


# scikit-util
What began as a succinct set of [sklearn](https://github.com/scikit-learn/scikit-learn) extension classes and utilities (as well as implementations of preprocessors from R packages like [caret](https://github.com/topepo/caret)) grew to bridge functionality between sklearn and [H2O](https://github.com/h2oai/h2o-3).  Now, scikit-util (skutil) brings the best of both worlds to H2O and sklearn, delivering an easy transition into the world of distributed computing that H2O offers, while providing the same, familiar interface that sklearn users have come to know and love.


### Installation:
```bash
git clone https://github.com/tgsmith61591/skutil.git
cd skutil
python setup.py install
```

#### Testing:
```bash
git clone https://github.com/tgsmith61591/skutil.git
cd skutil
python setup.py develop
nosetests
```


#### Troubleshooting Installation Issues
Skutil depends on the ability to compile Fortran code. For different platforms, there are different ways to install `gcc`:
  - Mac OS (__note__: this can take a while):
```bash
brew install gcc
```

There is a bug in some setups that will still cause issues in symlinking the `gcc` files via homebrew. If this is the case, the following line should clear things up:
```bash
brew link --overwrite gcc
```

  - Linux:
```bash
sudo apt-get install gcc
```

  - For Windows, follow [this tutorial](http://www.preshing.com/20141108/how-to-install-the-latest-gcc-on-windows/)


#### Examples:
  - See the [wiki](https://github.com/tgsmith61591/skutil/wiki)
  - See the [example ipython notebooks](https://github.com/tgsmith61591/skutil/tree/master/doc/examples)

