H2OPipeline
===========

The ``H2OPipeline`` generates a sklearn-esque pipeline of H2O steps finished with an optional ``H2OEstimator``. Note that as of version 0.1.0, the behavior of the H2OPipeline has slightly changed, given the inclusion of the ``exclude_from_ppc`` and ``exclude_from_fit`` parameters.

The pipeline, at the core, is comprised of a list of length-two tuples in the form of ``('name', SomeH2OTransformer())``, punctuated with an optional ``H2OEstimator`` as the final step. The pipeline will procedurally 
fit each stage, transforming the training data prior to fitting the next stage. When predicting or transforming new (test) data, each stage calls either `transform` or `predict` at the respective step.

**On the topic of exclusions and** ``feature_names``:

Prior to version 0.1.0, ``H2OTransformer`` classes did not take the keyword ``exclude_features``. Its addition necessitated two new keywords in the ``H2OPipeline``, and a slight change in behavior of ``feature_names``:

- ``exclude_from_ppc`` - If set in the ``H2OPipeline`` constructor, these features will be universally omitted from every preprocessing stage. Since ``exclude_features`` can be set individually in each separate transformer, in the case that ``exclude_features`` has been explicitly set, the exclusions in that respective stage will include the union of ``exclude_from_ppc`` and ``exclude_features``.


- ``exclude_from_fit`` - If set in the ``H2OPipeline`` constructor, these features will be omitted from the ``training_cols_`` fit attribute, which are the columns passed to the final stage in the pipeline.


- ``feature_names`` - The former behavior of the ``H2OPipeline`` only used ``feature_names`` in the fit of the first transformer, passing the remaining columns to the next transformer as the ``feature_names`` parameter. The new behavior is more discriminating in the case of explicitly-set attributes. In the case where a transformer's ``feature_names`` parameter has been explicitly set, *only those names* will be used in the fit. This is useful in cases where someone may only want to, for instance, drop one of two multicollinear features using the ``H2OMulticollinearityFilterer`` rather than fitting against the entire dataset. It also adheres to the now expected behavior of the exclusion parameters.


Here is a demo using sklearn's Boston Housing dataset.

.. code-block:: python

    from skutil.h2o import load_boston_h2o
    from skutil.h2o import h2o_train_test_split

    X = load_boston_h2o(include_tgt=True, shuffle=True, tgt_name='target')
    X_train, X_test = h2o_train_test_split(X, train_size=0.7) # this splits our data

    X_train.head() # it should look like:
    """
    | CRIM     | ZN | INDUS | CHAS | NOX   |  RM   | AGE  | DIS    | RAD | TAX | PTRATIO | B      | LSTAT | target|
    |----------|----|-------|------|-------|-------|------|--------|-----|-----|---------|--------|-------|-------|
    | 0.17783  | 0  | 9.69  | 0    | 0.585 | 5.569 | 73.5 | 2.3999 | 6   | 391 | 19.2    | 395.77 | 15.1  | 17.5  |
    | 6.80117  | 0  | 18.1  | 0    | 0.713 | 6.081 | 84.4 | 2.7175 | 24  | 666 | 20.2    | 396.9  | 14.7  | 20    |
    | 0.08707  | 0  | 12.83 | 0    | 0.437 | 6.14  | 45.8 | 4.0905 | 5   | 398 | 18.7    | 386.96 | 10.27 | 20.8  |
    """

    from skutil.h2o import H2OPipeline
    from skutil.h2o.transform import H2OSelectiveScaler
    from skutil.h2o.select import H2OMulticollinearityFilterer
    from h2o.estimators import H2OGradientBoostingEstimator

    # Declare our pipe - this one is intentionally a bit complex in behavior
    pipe = H2OPipeline([
            ('scl', H2OSelectiveScaler(feature_names=['B','PTRATIO','CRIM'])), # will ONLY operate on these features
            ('mcf', H2OMulticollinearityFilterer(exclude_features=['CHAS'])),  # will exclude this AS WELL AS 'TAX'
            ('gbm', H2OGradientBoostingEstimator())
        ],
        
        exclude_from_ppc=['TAX'], # excluded from all preprocessor fits
        feature_names=None,       # fit the first stage on ALL features (minus exceptions)
        target_feature='target')  # will be excluded from all preprocessor fits, as it's the target

    # do actual fit:
    pipe.fit(X_train)