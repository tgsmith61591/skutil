from __future__ import print_function, division
import warnings
import numpy as np
import pandas as pd
import abc
from pkg_resources import parse_version

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import six

import h2o
from h2o.frame import H2OFrame

from ..utils import is_numeric
from ..feature_selection import filter_collinearity
from .base import NAWarning


__all__ = [
    'BaseH2OTransformer',
    'H2OMulticollinearityFilterer'
]


def _check_is_frame(X):
    """Returns X if X is a frame else throws a TypeError"""
    if not isinstance(X, H2OFrame):
        raise TypeError('expected H2OFrame but got %s' % type(X))
    return X

def _retain_features(X, exclude):
    """Returns the features to retain"""
    return [x for x in X.columns if not x in exclude]


class BaseH2OTransformer(BaseEstimator, TransformerMixin):
    """Base class for all H2OTransformers.

    Parameters
    ----------
    target_feature : str (default None)
        The name of the target feature (is excluded from the fit)

    min_version : str, float (default 'any')
        The minimum version of h2o that is compatible with the transformer
    """
    
    @abc.abstractmethod
    def __init__(self, target_feature=None, min_version='any'):
        
        # validate version
        h2ov = h2o.__version__
                
        # won't enter this block if passed at 'any'
        if is_numeric(min_version): # then int or float
            min_version = str(min_version)
        
        if isinstance(min_version, str):
            if min_version == 'any':
                pass # anything goes
            else:
                if parse_version(h2ov) < parse_version(min_version):
                    raise EnvironmentError('your h2o version (%s) '
                                           'does not meet the minimum ' 
                                           'requirement for this transfromer (%s)'
                                           % (h2ov, str(min_version)))
        
        else:
            raise ValueError('min_version must be a float, '
                             'a string in the form of "X.x" '
                             'or "any", but got %s' % type(min_version))
            
    @property
    def min_version(self):
        try:
            return self.__min_version__
        except NameError as n:
            return 'any'


class H2OMulticollinearityFilterer(BaseH2OTransformer):
    """Filter out features with a correlation greater than the provided threshold.
    When a pair of correlated features is identified, the mean absolute correlation (MAC)
    of each feature is considered, and the feature with the highsest MAC is discarded.

    Parameters
    ----------
    target_feature : str (default None)
        The name of the target feature (is excluded from the fit)
    
    threshold : float, default 0.85
        The threshold above which to filter correlated features
        
    na_warn : bool (default True)
        Whether to warn if any NAs are present

    Attributes
    ----------
    drop : list, string
        The columns to drop
    """
    
    __min_version__ = 3.8
    
    def __init__(self, target_feature=None, threshold=0.85, na_warn=True):
        super(H2OMulticollinearityFilterer, self).__init__(target_feature, 
                                                           self.__min_version__)
        self.threshold = threshold
        self.na_warn = na_warn
        
        
    def fit(self, X, y=None):
        """Fit the multicollinearity filterer.

        Parameters
        ----------
        X : H2OFrame
            The frame to fit

        y : None, passthrough for pipeline
        """

        self.fit_transform(X, y)
        return self
    
    
    def fit_transform(self, X, y=None):
        """Fit the multicollinearity filterer and
        return the transformed H2OFrame, X.

        Parameters
        ----------
        X : H2OFrame
            The frame to fit

        y : None, passthrough for pipeline
        """
        
        frame, thresh = _check_is_frame(X), self.threshold
        
        # if there's a target feature, let's strip it out for now...
        if self.target_feature:
            X_nms = [x for x in frame.columns if not x == self.target_feature] # make list
            frame = frame[X_nms]
            
        # check on NAs
        if self.na_warn:
            nasum = X.isna().sum()
            if nasum > 0:
                warnings.warn('%i NA value(s) in frame; using "complete.obs"' % nasum)
        
        ## Generate absolute correlation matrix
        c = frame.cor(use='complete.obs').abs().as_data_frame(use_pandas=True)
        
        ## get drops list
        self.drop = filter_collinearity(c, self.threshold)
        retain = _retain_features(X, self.drop) # pass original frame

        return frame[retain]
        
        
    def transform(self, X):
        X = _check_is_frame(X)
        retain = _retain_features(X, self.drop)
        return X[retain]