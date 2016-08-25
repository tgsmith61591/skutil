
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import column_or_1d
from sklearn.preprocessing.label import _check_numpy_unicode_bug
import numpy as np
import pandas as pd
from skutil.utils import validate_is_pd


__all__ = [
    'SafeLabelEncoder',
    'OneHotCategoricalEncoder'
]


def get_unseen():
    """Basically just a static method
    instead of a class attribute to avoid
    someone accidentally changing it."""
    return 99999


class SafeLabelEncoder(LabelEncoder):
    """An extension of LabelEncoder that will
    not throw an exception for unseen data, but will
    instead return a default value of 99999
    
    Attributes
    ----------
    classes_ : the classes that are encoded
    """
        
    def transform(self, y):
        check_is_fitted(self, 'classes_')
        y = column_or_1d(y, warn=True)
        
        classes = np.unique(y)
        _check_numpy_unicode_bug(classes)
        
        ## Check not too many:
        unseen = get_unseen()
        if len(classes) >= unseen:
            raise ValueError('Too many factor levels in feature. Max is %i' % unseen)
        
        return np.array([np.searchsorted(self.classes_, x)\
                         if x in self.classes_\
                         else unseen\
                         for x in y])


class OneHotCategoricalEncoder(BaseEstimator, TransformerMixin):
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

    as_df : boolean, default = True
        Whether to return a pandas dataframe
        
    Attributes
    ----------
    fill : see above

    as_df : see above
    
    obj_cols_ : array_like
        The list of object-type (categorical) features

    lab_encoders_ : array_like
        The label encoders

    one_hot_ : an instance of a OneHotEncoder
    
    trans_nms_ : the dummified names
    """
    
    def __init__(self, fill='Missing', as_df=True):
        self.fill = fill
        self.as_df = as_df
        
        
    def fit(self, X, y = None):
        """Fit the estimator.
        
        Parameters
        ----------
        X : pandas dataframe
        y : passthrough for Pipeline
        """
        # check on state of X, don't care about cols or the warning
        X, _ = validate_is_pd(X, None)
            
        ## Extract the object columns
        obj_cols_ = X.select_dtypes(include = ['object']).columns.values
        
        ## If we need to fill in the NAs, take care of it
        if not self.fill is None:
            X[obj_cols_] = X[obj_cols_].fillna(self.fill)
        
        ## Set an array of uninitialized label encoders
        ## Then use fit_transform for effiency purposes
        ## We can also set the dummy-level feature names in the same pass
        lab_encoders_ = []
        trans_array = []
        tnms = []
        
        unseen = get_unseen()
        for nm in obj_cols_:
            encoder = SafeLabelEncoder()
            lab_encoders_.append(encoder)
            
            ## This fits the reference to the encoder, and gets
            ## the transformation. We then append a single unseen
            ## value to the end as a safety for the transform method.
            ## After the transpose, this is tantamount to appending a row
            ## of unseen values so each feature can handle the 99999
            ## This will expand the matrix by N columns, but if there's 
            ## no new values, they will be entirely zero and can be dropped later.
            encoded_array = np.append(encoder.fit_transform(X[nm]), unseen)
            
            ## Add the transformed row
            trans_array.append(encoded_array) ## Updates in array
            
            ## Update the names
            n_classes = len(encoder.classes_)
            sequential_nms = ['%s.%s' % (nm,str(encoder.classes_[i])) for i in range(n_classes)]
            
            ## Remember to append the NA col
            sequential_nms.append('%s.NA' % nm)
            tnms.append(sequential_nms)
        
        ## Get the transpose
        trans = np.array(trans_array).transpose()
            
        ## flatten the name array, append numeric names prior
        num_nms = [n for n in X.columns.values if not n in obj_cols_]
        trans_nms_= [item for sublist in tnms for item in sublist]
        self.trans_nms_ = num_nms + trans_nms_
        
        # we might get an empty set of object cols
        shape_tup = trans.shape
        is_empty = len(shape_tup) < 2 or shape_tup[1] == 0 # zero cols

        ## Now we can do the actual one hot encoding, set internal state
        self.one_hot_ = None if is_empty else OneHotEncoder().fit(trans)
        self.obj_cols_ = obj_cols_
        self.lab_encoders_ = lab_encoders_
        
        return self
        
    def transform(self, X):
        """Transform X, a DataFrame, by stripping
        out the object columns, dummifying them, and
        re-appending them to the end.
        
        Parameters
        ----------
        X : pandas dataframe
        """
        check_is_fitted(self, 'obj_cols_')
        # check on state of X, don't care about cols or warning
        X, _ = validate_is_pd(X, None)

        # if there is no encoder to speak of, just bail early
        if not self.one_hot_:
            return X if self.as_df else X.as_matrix()
        
        ## Retain just the numers
        numers = X[[nm for nm in X.columns.values if not nm in self.obj_cols_]]
        objs = X[self.obj_cols_]
        
        ## If we need to fill in the NAs, take care of it
        if not self.fill is None:
            objs = objs.fillna(self.fill)
            
        ## Do label encoding using the safe label encoders
        trans = np.array([v.transform(objs[self.obj_cols_[i]]) for\
                          i,v in enumerate(self.lab_encoders_)]).transpose()
        
        ## Finally, get the one-hot encoding...
        oh = self.one_hot_.transform(trans).todense()
        x = np.array(np.hstack((numers, oh)))

        return x if not self.as_df else pd.DataFrame.from_records(data=x, columns=self.trans_nms_)

