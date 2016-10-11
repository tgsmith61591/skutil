Encoders
=========
Currently implemented encoders:

- ``OneHotCategoricalEncoder``:
  - Should be the first phase in your ``Pipeline`` object. Takes a Pandas dataframe, imputes missing categorical data with a provided string and dummies out the object (string) columns. Finally, returns a ``numpy.ndarray`` transformed array.
- ``SafeLabelEncoder``:
  - Wraps sklearn's ``LabelEncoder``, but encodes unseen data in your test set as a default factor-level value (99999).

.. code-block:: python

   ## Example use of OneHotCategoricalEncoder
   import numpy as np
   from skutil.preprocessing import SafeLabelEncoder, OneHotCategoricalEncoder
   import pandas as pd

   ## An array of strings
   X = np.array([['USA','RED','a'],
                 ['MEX','GRN','b'],
                 ['FRA','RED','b']])
   x = pd.DataFrame.from_records(data = X, columns = ['A','B','C'])

   ## Tack on a numeric col:
   x['n'] = np.array([5,6,7])

   ## Fit the encoder -- default return is pandas DataFrame
   o = OneHotCategoricalEncoder(as_df=False).fit(x)

   ## Notice that the numeric data is now BEFORE the dummies
   >>> o.transform(x)
   [[ 5.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  0.,  0.],
    [ 6.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.],
    [ 7.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.]]

   ## We can extract the new names:
   >>> o.trans_nms_
   ['n', 'A.FRA', 'A.MEX', 'A.USA', 'A.NA', 'B.GRN', 'B.RED', 'B.NA', 'C.a', 'C.b', 'C.NA']

   ## Notice we have one extra factor level for each column (i.e., 'A.NA').
   ## This is to hold factor levels in testing that we didn't see in training.
   ## Most sklearn algorithms will shrink that coefficient to zero in training,
   ## or completely ignore it so it's merely a placeholder for elegant handling
   ## of new data. Let's test what happens on unseen data:
   Y = np.array([['CAN','BLU','c']])
   y = pd.DataFrame.from_records(data = Y, columns = ['A','B','C'])

   ## Add the numeric var in at the end
   y['n'] = np.array([7])
   >>> o.transform(y)
   [[ 7.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.]]

   ## Notice only the 'x.NA' features are populated!