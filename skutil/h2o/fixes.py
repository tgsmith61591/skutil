from __future__ import print_function, absolute_import, division
import h2o
from h2o.frame import H2OFrame
from h2o.expr import ExprNode
import numpy as np
from pkg_resources import parse_version
from .base import check_frame

_h2ov = h2o.__version__

__all__ = [
    'rbind_all'
]

if parse_version(_h2ov) < parse_version('3.10.0.7'):
    def rbind_all(*args):
        """Given a variable set of H2OFrames,
        rbind all of them into a single H2OFrame.

        Parameters
        ----------

        array1, array2, ... : H2OFrame, shape=(n_samples, n_features)
            The H2OFrames to rbind. All should match in column
            dimensionality.


        Returns
        -------

        f : H2OFrame
            The rbound H2OFrame
        """
        # check all are H2OFrames
        for x in args:
            check_frame(x, copy=False)

        # check col dim
        if np.unique([x.shape[1] for x in args]).shape[0] != 1:
            raise ValueError('inconsistent column dimensions')

        f = None
        for x in args:
            f = x if f is None else f.rbind(x)

        return f

else:
    def rbind_all(*args):
        """Given a variable set of H2OFrames,
        rbind all of them into a single H2OFrame.

        Parameters
        ----------

        array1, array2, ... : H2OFrame, shape=(n_samples, n_features)
            The H2OFrames to rbind. All should match in column
            dimensionality.


        Returns
        -------

        f : H2OFrame
            The rbound H2OFrame
        """
        # lazily evaluate type on the h2o side
        if len(args) == 1:
            return args[0]

        def rbind(*data):
            slf = data[0]
            nrow_sum = 0

            for frame in data:
                if frame.ncol != slf.ncol:
                    raise ValueError("Cannot row-bind a dataframe with %d columns to a data frame with %d columns: "
                                     "the columns must match" % (frame.ncol, slf.ncol))
                if frame.columns != slf.columns or frame.types != slf.types:
                    raise ValueError("Column names and types must match for rbind() to work")
                nrow_sum += frame.nrow

            fr = H2OFrame._expr(expr=ExprNode("rbind", slf, *data[1:]), cache=slf._ex._cache)
            fr._ex._cache.nrows = nrow_sum
            return fr

        return rbind(*args)
