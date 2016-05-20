import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

__all__ = [
	'get_numeric',
	'is_numeric',
	'perfect_collinearity_test',
	'validate_is_pd'
]

def validate_is_pd(X):
    if not isinstance(X, pd.DataFrame):
        raise ValueError('expected pandas DataFrame')


def get_numeric(X):
    """Return list of indices of numeric dtypes variables

    Parameters
    ----------
    X : pandas DF
        The dataframe
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError('expected pandas DF')

    return X.dtypes[X.dtypes.apply(lambda x: str(x).startswith(("float", "int", "bool")))].index.tolist()


def is_numeric(x):
	"""Determines whether the X is numeric

    Parameters
    ----------
    x : anytype
    """
	return isinstance(x, (int, float, long, np.int, np.float, np.long))


def perfect_collinearity_test(X, min_rows="infer", max_rows=None):
    """Test input data for any perfect correlations by running a regression
    against every x on all other x features. This is adaptive; it starts with
    a small dataset and if any R^2 == 1, will double the size of the dataset
    ad nauseum.


    Parameters:
    -----------
    X : a pandas dataframe

    max_rows : Most rows the model will use for a variable
    """
    # Sets the minimum number of rows to start with.
    if min_rows == "infer":
        rows_to_use = 2*X.shape[1]
        if rows_to_use > X.shape[0]:
            rows_to_use = X.shape[0]
    else:
    	if not is_numeric(X):
    		raise ValueError('expected numeric for min_rows')
        rows_to_use = min_rows
    

    # Sets the maximum number of rows to use.
    if max_rows is None:
        max_rows = X.shape[0]

	
	max_rows = np.minimum(max_rows, X.shape[0]) ## ensure not too many...    
    numeric_cols = get_numeric(X)

    ## ensure at least 2
    if len(numeric_cols) < 2:
        raise ValueError('fewer than 2 numeric columns in X!')

    columns_in_dataframe = X[numeric_cols].columns ## Only the numeric columns


    
    # Series to save results
    results = pd.Series(name = 'R^2')
    
    # Runs a regression of every x against all other X variables.
    # Starts with a small dataset and if R^2 == 1, doubles the size
    # of the dataset until greater than max_rows.
    for temp_y_variable in columns_in_dataframe:
        rows_to_use_base = rows_to_use

        while True:
            X_master = X[:rows_to_use_base]
            temp_X_variables = [col for col in columns_in_dataframe if not col == temp_y_variable]

            y_temp = X_master[temp_y_variable]
            X_temp = X_master[temp_X_variables]

            lin_model = LinearRegression().fit(X_temp, y_temp)
            R_2 = lin_model.score(X_temp, y_temp)

            if R_2 != 1 and R_2 >= 0 or rows_to_use_base >= max_rows:
                results[temp_y_variable] = R_2
                break

            ## Double the size but not too large
            rows_to_use_base *= 2
            rows_to_use_base = np.minimum(rows_to_use_base, X.shape[0])

    return results
