"""Data augmentation inspired by mixup: beyond empirical risk minimization."""

# imports
# constants
# exception classes
# interface functions
# classes
# internal functions & classes

import sys
import random
import numpy as np
import pandas as pd


def mixup(data, alpha=4, concat=False, batch_size=None):
    """
    Create convex combinations of pairs of examples and their labels
    for data augmentation and regularisation

    This function enlarges training sets using linear interpolations of
    features and associated labels as described in
    https://arxiv.org/abs/1710.09412.

    The data must be numeric.  Non-finite values are not permitted.
    Factors should be one-hot encoded.  Duplicate values will not
    be removed.

    For now, only binary classification is supported.  Meaning the y
    coloumn must contain only numeric 0 and 1 values.

    Alpha values must be greater than or equal to zero.  Alpha equal to
    zero specifies no interpolation.

    The mixup function returns a pandas dataframe containing interpolated
    x and y values.  Optionally, the original values can be concatenated
    with the new values.

    Parameters
    __________
    data : pandas dataframe
      Original features and labels
    alpha : float, optional
      Hyperparameter specifying strength of interpolation
    concat : bool, optional
      Concatenate mixup data with original data
    batch_size : int, optional
      How many mixup values to produce

    Returns
    _______
    A pandas dataframe containing interpolated x and y values and
    optionally the original values

    Examples
    ________
    >>> data_mix = mixup(data, 'y')

    See also
    ________
    https://github.com/makeyourownmaker/mixupy
    """

    _check_data(data)
    _check_params(alpha, concat, batch_size)

    data_len = data.shape[0]

    if batch_size is None:
        batch_size = data_len

    # Used to shuffle data2
    if batch_size <= data_len:
        # no replacement
        index = random.sample(range(0, data_len), batch_size)
    else:
        # with replacement
        index = np.random.randint(0, data_len, size=batch_size)

    data_orig = data

    # Make data1 same size as data2
    data1 = resize_data(data, batch_size)

    data2 = data1.loc[index]
    data2 = data2.reset_index(drop=True)

    # x <- lam * x1 + (1. - lam) * x2
    # y <- lam * y1 + (1. - lam) * y2
    lam = np.random.beta(alpha, alpha, size=(batch_size, 1))
    data_mix = lam * data1 + (1.0 - lam) * data2

    data_new = data_mix

    if concat is True:
        data_new = pd.concat([data_orig, data_mix])

    data_new.columns = data_orig.columns

    return data_new


def resize_data(data, batch_size):
    """Resize data by repeating/removing rows"""

    data_orig = data
    data_len = data.shape[0]

    if data_len < batch_size:
        rep_times = batch_size // data_len

        for _ in range(rep_times):
            data = pd.concat([data, data_orig])

        data = data.reset_index(drop=True)

    if data_len < batch_size:
        data = data.loc[: batch_size - 1, :]
    else:
        data = data.loc[:batch_size, :]

    return data


def printe(errmsg):
    """Print error message and exit"""

    print(errmsg)
    sys.exit(1)


def _check_data_is_numeric(data):
    """Check data is numeric (int or float)"""

    # numerics = data.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())
    numerics = data.shape[1] == data.select_dtypes(include=np.number).shape[1]

    if numerics is False:
        errmsg = (
            "Values must be numeric in 'data':\n"
            + " non-numeric values found\n"
            + str(data.dtypes)
        )
        printe(errmsg)

    return 0


def _check_data_is_finite(data):
    """Check data is finite - no NAs and no infs"""

    errmsg = "Values must be finite in 'data':\n"
    nas = pd.isna(data).sum()

    if np.sum(nas) > 0:
        errmsg += " 'na's found at \n" + str(nas)
        printe(errmsg)

    # infs = np.isinf(data).sum()
    infs = np.isinf(data.select_dtypes(include=np.number)).sum()

    if np.sum(infs) > 0:
        errmsg += " 'inf's found at\n" + str(infs)
        printe(errmsg)

    return 0


def _check_data(data):

    if not isinstance(data, pd.DataFrame):
        errmsg = "'data' must be pandas dataframe.\n" + "  'data' is ", type(data), "\n"
        printe(errmsg)

    if data.shape[0] < 2:
        errmsg = (
            "'data' must have 2 or more rows.\n" + "  'data' has ",
            data.shape[0],
            " rows.\n",
        )
        printe(errmsg)

    if data.shape[1] < 2:
        errmsg = (
            "'data' must have 2 or more columns.\n" + "  'data' has ",
            data.shape[1],
            " columns.\n",
        )
        printe(errmsg)

    _check_data_is_numeric(data)
    _check_data_is_finite(data)

    return 0


def _check_params(alpha, concat, batch_size):

    if not isinstance(alpha, (int, float)):
        errmsg = "'alpha' must be integer or float\n" + "  'alpha' is ", alpha, "\n"
        printe(errmsg)

    if alpha < 0:
        errmsg = (
            "'alpha' must be greater than or equal to 0.\n" + "  'alpha' is ",
            alpha,
            "\n",
        )
        printe(errmsg)

    if not isinstance(concat, bool):
        errmsg = "'concat' must be True or False:\n" + "  'concat' is ", concat, "\n"
        printe(errmsg)

    if batch_size is not None and not isinstance(batch_size, int):
        errmsg = (
            "'batch_size' must be an integer\n" + "  'batch_size' is ",
            batch_size,
            "\n",
        )
        printe(errmsg)

    if batch_size is not None and batch_size <= 0:
        errmsg = (
            "'batch_size' must be greater than 0.\n" + "  'batch_size' is ",
            batch_size,
            "\n",
        )
        printe(errmsg)

    return 0
