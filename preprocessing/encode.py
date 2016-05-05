from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import column_or_1d
from sklearn.preprocessing.label import _check_numpy_unicode_bug
import numpy as np
import pandas as pd


__all__ = [
    'SafeLabelEncoder',
    'OneHotCategoricalTransformer'
]


class SafeLabelEncoder(LabelEncoder):
    """An extension of LabelEncoder that will
    not throw an exception for unseen data, but will
    instead return a default value
    
    Attributes
    ----------
    __default__ : int, default val = 999999
    """
    __default__ = 999999
        
    def transform(self, y):
        check_is_fitted(self, 'classes_')
        y = column_or_1d(y, warn=True)
        
        classes = np.unique(y)
        _check_numpy_unicode_bug(classes)
        
        return np.array([np.searchsorted(self.classes_, x)\
                         if x in self.classes_\
                         else self.__default__\
                         for x in y])


class OneHotCategoricalTransformer(BaseEstimator, TransformerMixin):
    """This class achieves three things: first, it will fill in 
    any NaN values with a provided surrogate (if desired). Second,
    it will dummy out any categorical features using OneHotEncoding
    with a safety feature that can handle previously unseen values,
    and in the transform method will re-append the dummified features
    to the dataframe. Finally, it will return a numpy ndarray.
    
    Parameters
    ----------
    fill : str, optional (default = 'Missing')
        The value that will fill the missing values in the column
        
    Attributes
    ----------
    fill_ : see above
    
    self.__unseen__ : int, default = 999999
        If transforming data encounters a value it hasn't seen before,
        it will set it to this
    
    obj_cols_ : array_like
        The list of object-type (categorical) features
    lab_encoders_ : array_like
        The label encoders
    one_hot_ : an instance of a OneHotEncoder
    trans_nms_ : the dummified names
    """
    
    __unseen__ = 999999
    
    def __init__(self, fill = 'Missing'):
        self.fill_ = fill
        
        
    def fit(self, X, y = None):
        """Fit the estimator.
        
        Parameters
        ----------
        X : pandas dataframe
        y : passthrough for Pipeline
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError('expected Pandas DataFrame')
            
        ## Extract the object columns
        X = X.copy()
        obj_cols_ = X.select_dtypes(include = ['object']).columns.values
        
        ## If we need to fill in the NAs, take care of it
        if not self.fill_ is None:
            X = X[obj_cols_].fillna(self.fill_)
        
        ## Get array of label encoders
        lab_encoders_ = [SafeLabelEncoder().fit(X[nm]) for nm in obj_cols_]

        ## Check no factor levels over 999999
        for encoder in lab_encoders_:
            if len(encoder.classes_) >= self.__unseen__:
                raise ValueError('too many factor levels in feature')
        
        ## Use the fit encoders to get the encoded matrix
        trans = np.array([v.transform(X[obj_cols_[i]]) for\
                          i,v in enumerate(lab_encoders_)]).transpose()
        
        ## We add a row of the unseen values so the onehotencoder
        ## has seen the unseen values before, and won't throw an
        ## exception later with new data. This will expand the matrix
        ## by N columns, but if there's no new values, they will be
        ## entirely zero and can be dropped later.
        trans = np.vstack((trans, np.ones(trans.shape[1]) * self.__unseen__))
        
        ## Set the trans, dummy-level feature names
        tnms = []
        for i,v in enumerate(lab_encoders_):
            nm = obj_cols_[i]
            n_classes = len(set(v.classes_))
            tnms.append(['%s.%i' % (nm,i) for i in range(n_classes)])
            
        ## flatten that array
        self.trans_nms_= reduce(lambda i,j: i + j, tnms)
        
        ## Now we can do the actual one hot encoding, set internal state
        self.one_hot_ = OneHotEncoder().fit(trans)
        self.obj_cols_ = obj_cols_
        self.lab_encoders_ = lab_encoders_
        
        return self
        
        
    def transform(self, X, y = None):
        """Transform X, a DataFrame, by stripping
        out the object columns, dummifying them, and
        re-appending them to the end.
        
        Parameters
        ----------
        X : pandas dataframe
        y : passthrough for Pipeline
        """
        check_is_fitted(self, 'obj_cols_')
        if not isinstance(X, pd.DataFrame):
            raise ValueError('expected Pandas DataFrame')
        
        X = X.copy()
        
        ## Retain just the numers
        numers = X[[nm for nm in X.columns.values if not nm in self.obj_cols_]]
        objs = X[self.obj_cols_]
        
        ## If we need to fill in the NAs, take care of it
        if not self.fill_ is None:
            objs = objs.fillna(self.fill_)
            
        ## Do label encoding using the safe label encoders
        trans = np.array([v.transform(objs[self.obj_cols_[i]]) for\
                          i,v in enumerate(self.lab_encoders_)]).transpose()
        
        ## Finally, get the one-hot encoding...
        oh = self.one_hot_.transform(trans).todense()
        return np.array(np.hstack((numers, oh)))
