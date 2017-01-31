from __future__ import print_function, absolute_import, division
import h2o
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
        if isinstance(args, (tuple, list)):
            lst = args[0]
            if len(lst) == 1:  # there's only one element
                return lst[0]
            return lst[0].rbind(lst[1:])
        if len(args) == 1:
            return args[0]
        return args[0].rbind(args[1:])
