import pytest
import numpy as np
import pandas as pd
from mixupy import mixup


@pytest.fixture
def example_data_frame():
    '''dataframe for testing'''

    rows = 5
    cols = 5
    limit = 16

    return pd.DataFrame(np.random.uniform(-limit, limit, size=(rows, cols)))


@pytest.fixture
def example_data_array():
    '''dataframe for testing'''

    rows = 5
    cols = 5
    limit = 16

    return np.array(np.random.uniform(-limit, limit, size=(rows, cols)))


@pytest.fixture
def example_nan_data_frame(example_data_frame):
    '''dataframe for testing with nan values'''

    df = example_data_frame
    df.loc[0, 0] = np.nan
    df.loc[1, 1] = np.nan

    return df


@pytest.fixture
def example_inf_data_frame(example_data_frame):
    '''dataframe for testing with inf values'''

    df = example_data_frame
    df.loc[0, 0] = np.inf
    df.loc[1, 1] = np.inf

    return df


# E       TypeError: ufunc 'isinf' not supported for the input types, and the inputs could
# not be safely coerced to any supported types according to the casting rule ''safe''
@pytest.fixture
def example_nonnumeric_data_frame(example_data_frame):
    '''dataframe for testing with non-numeric values'''

    df = example_data_frame
    df.loc[0, 0] = 'foo'
    df.loc[1, 1] = 'bar'
    # also tried 'foo','bar' and True,False and '1','2' and '1.0','2.0'
    # all gave the above TypeError

    return df


@pytest.fixture
def example_one_row_data_frame():
    '''Single row dataframe for testing'''

    return pd.DataFrame([[0, 1]], columns=['A', 'B'])


@pytest.fixture
def example_one_column_data_frame():
    '''Single column dataframe for testing'''

    return pd.DataFrame([[0], [1]], columns=['A'])


def test_default_mixup(example_data_frame):
    '''mixup with default settings'''
    assert mixup(example_data_frame).shape == example_data_frame.shape


# E       TypeError: ufunc 'isinf' not supported for the input types, and the inputs could
# not be safely coerced to any supported types according to the casting rule ''safe''
def test_nonnumeric_data_frame(example_nonnumeric_data_frame):
    '''mixup with non-numeric values in dataframe'''
    with pytest.raises(SystemExit) as e:
        mixup(example_nonnumeric_data_frame)
    assert e.type == SystemExit
    assert e.value.code == 1


def test_concat_mixup(example_data_frame):
    '''mixup with concat'''
    assert mixup(example_data_frame, concat=True).shape[0] == example_data_frame.shape[0] * 2
    assert mixup(example_data_frame).shape[1] == example_data_frame.shape[1]


def test_data_resize(example_data_frame):
    '''test the data_resize function'''
    bs = example_data_frame.shape[0] * 2
    assert mixup(example_data_frame, batch_size=bs).shape[0] == example_data_frame.shape[0] * 2
    assert mixup(example_data_frame, batch_size=bs).shape[1] == example_data_frame.shape[1]


def test_nonbool_alpha(example_data_frame):
    '''mixup with non-bool concat'''
    with pytest.raises(SystemExit) as e:
        mixup(example_data_frame, concat='True')
    assert e.type == SystemExit
    assert e.value.code == 1


def test_nonnumeric_alpha(example_data_frame):
    '''mixup with non-numeric alpha'''
    with pytest.raises(SystemExit) as e:
        mixup(example_data_frame, alpha='one')
    assert e.type == SystemExit
    assert e.value.code == 1


def test_negative_alpha(example_data_frame):
    '''mixup with negative alpha'''
    with pytest.raises(SystemExit) as e:
        mixup(example_data_frame, alpha=-1)
    assert e.type == SystemExit
    assert e.value.code == 1


def test_nonnumeric_batch_size(example_data_frame):
    '''mixup with non-numeric batch_size'''
    with pytest.raises(SystemExit) as e:
        mixup(example_data_frame, batch_size='one')
    assert e.type == SystemExit
    assert e.value.code == 1


def test_negative_batch_size(example_data_frame):
    '''mixup with negative batch_size'''
    with pytest.raises(SystemExit) as e:
        mixup(example_data_frame, batch_size=-1)
    assert e.type == SystemExit
    assert e.value.code == 1


def test_nan_exit(example_nan_data_frame):
    '''mixup with dataframe containing nan values'''
    with pytest.raises(SystemExit) as e:
        mixup(example_nan_data_frame)
    assert e.type == SystemExit
    assert e.value.code == 1


def test_inf_exit(example_inf_data_frame):
    '''mixup with dataframe containing inf values'''
    with pytest.raises(SystemExit) as e:
        mixup(example_inf_data_frame)
    assert e.type == SystemExit
    assert e.value.code == 1


def test_array(example_data_array):
    '''mixup with array containing'''
    with pytest.raises(SystemExit) as e:
        mixup(example_data_array)
    assert e.type == SystemExit
    assert e.value.code == 1


def test_one_row_data_frame(example_one_row_data_frame):
    '''mixup with dataframe containing a single row'''
    with pytest.raises(SystemExit) as e:
        mixup(example_one_row_data_frame)
    assert e.type == SystemExit
    assert e.value.code == 1


def test_one_column_data_frame(example_one_column_data_frame):
    '''mixup with dataframe containing a single column'''
    with pytest.raises(SystemExit) as e:
        mixup(example_one_column_data_frame)
    assert e.type == SystemExit
    assert e.value.code == 1
