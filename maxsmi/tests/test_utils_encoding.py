"""
Unit and regression test for the maxsmi package.
"""

# Import package, test suite, and other packages as needed
import pytest
import sys
import numpy as np
from maxsmi.utils_encoding import (
    get_max_length,
    char_replacement,
    get_unique_elements_as_dict,
    one_hot_encode,
    pad_matrix,
)


def test_maxsmi_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "maxsmi" in sys.modules


####################
@pytest.mark.parametrize(
    "list_, solution",
    [
        (["1", "helloworld!", "12"], 11),
        (["a", "b", "CC1CC1"], 6),
    ],
)
def test_get_max_length(list_, solution):
    max_len = get_max_length(list_)
    assert solution == max_len


####################
@pytest.mark.parametrize(
    "smiles, solution",
    [("Cl", "L"), ("CBrCC", "CRCC"), ("C@@C", "C$C")],
)
def test_char_replacement(smiles, solution):
    replace_smiles = char_replacement(smiles)
    assert solution == replace_smiles


####################
@pytest.mark.parametrize(
    "list_, solution",
    [
        (["@", "B", "1"], {"1": 0, "@": 1, "B": 2}),
        (["2", "CCC"], {"2": 0, "C": 1}),
    ],
)
def test_get_unique_elements_as_dict(list_, solution):
    dict_ = get_unique_elements_as_dict(list_)
    assert solution == dict_


####################
@pytest.mark.parametrize(
    "smiles, solution",
    [
        ("C1", np.array([[0, 1], [1, 0]])),
    ],
)
def test_one_hot_encode(smiles, solution):
    dictionary = get_unique_elements_as_dict(smiles)
    one_hot_matrix = one_hot_encode(smiles, dictionary)
    assert (solution == one_hot_matrix).all()


####################
@pytest.mark.parametrize(
    "matrix, max_pad, solution",
    [(np.zeros((2, 2)), 4, np.zeros((2, 4))), (np.zeros((2, 2)), 1, np.zeros((2, 2)))],
)
def test_pad_matrix(matrix, max_pad, solution):
    padded_matrix = pad_matrix(matrix, max_pad)
    assert (solution == padded_matrix).all()
